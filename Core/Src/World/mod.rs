pub mod block;
pub mod chunk;
pub mod biomes;
pub mod generator;
pub mod raycast;
pub mod palette;

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use chunk::{Chunk, CHUNK_W, CHUNK_D};
use biomes::BiomeGenerator;
use generator::{ChunkGenerator, GeneratorMessage};
pub use block::BlockType;

pub struct World {
    pub chunks: HashMap<(i32, i32), Chunk>,
    generator: Arc<BiomeGenerator>,
    chunk_gen: ChunkGenerator,
    gen_receiver: mpsc::Receiver<GeneratorMessage>,
    pending_chunks: std::collections::HashSet<(i32, i32)>,
}

#[derive(Serialize, Deserialize)]
struct ChunkSave {
    cx: i32,
    cz: i32,
    blocks: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct WorldSave {
    seed: u32,
    chunks: Vec<ChunkSave>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        let gen = Arc::new(BiomeGenerator::new(seed));
        let (chunk_gen, gen_receiver) = ChunkGenerator::new_with_seed(seed);
        
        Self {
            chunks: HashMap::new(),
            generator: Arc::clone(&gen),
            chunk_gen,
            gen_receiver,
            pending_chunks: std::collections::HashSet::new(),
        }
    }

    /// Process any completed chunks from background generation
    ///
    /// Returns true if at least one chunk was inserted.
    pub fn process_generated_chunks(&mut self) -> bool {
        let mut inserted = false;
        // Non-blocking: try to receive all ready chunks
        while let Ok(GeneratorMessage::ChunkReady(cx, cz, chunk)) = self.gen_receiver.try_recv() {
            self.chunks.insert((cx, cz), chunk);
            self.pending_chunks.remove(&(cx, cz));
            inserted = true;
        }
        inserted
    }

    /// Queue chunks for loading within `radius` of chunk (cx, cz)
    pub fn load_around(&mut self, cx: i32, cz: i32, radius: i32) {
        // Ensure the current chunk and its immediate neighbors exist before background loading.
        // This avoids the player moving into unloaded space and falling through visible terrain.
        let sync_radius = 1;

        if self.chunks.is_empty() {
            for dz in -sync_radius..=sync_radius {
                for dx in -sync_radius..=sync_radius {
                    let key = (cx + dx, cz + dz);
                    if !self.chunks.contains_key(&key) {
                        let chunk = Chunk::generate(key.0, key.1, &self.generator);
                        self.chunks.insert(key, chunk);
                    }
                }
            }
        }

        let mut to_generate = Vec::new();
        
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let key = (cx + dx, cz + dz);
                if self.chunks.contains_key(&key) {
                    continue;
                }

                if dx.abs() <= sync_radius && dz.abs() <= sync_radius {
                    let chunk = Chunk::generate(key.0, key.1, &self.generator);
                    self.chunks.insert(key, chunk);
                } else if !self.pending_chunks.contains(&key) {
                    to_generate.push(key);
                    self.pending_chunks.insert(key);
                }
            }
        }
        
        // Queue all distant chunks for background generation
        if !to_generate.is_empty() {
            self.chunk_gen.queue_chunks(&to_generate);
        }
    }

    /// Unload chunks outside the given radius
    pub fn unload_far_chunks(&mut self, cx: i32, cz: i32, radius: i32) {
        let unload_radius = radius + 2; // buffer
        let mut to_remove = Vec::new();
        for &(chunk_x, chunk_z) in self.chunks.keys() {
            let dx = (chunk_x - cx).abs();
            let dz = (chunk_z - cz).abs();
            if dx > unload_radius || dz > unload_radius {
                to_remove.push((chunk_x, chunk_z));
            }
        }
        for key in to_remove {
            self.chunks.remove(&key);
            self.pending_chunks.remove(&key);
        }
    }

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&(cx, cz))
    }

    pub fn get_chunk_mut(&mut self, cx: i32, cz: i32) -> Option<&mut Chunk> {
        self.chunks.get_mut(&(cx, cz))
    }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> BlockType {
        if wy < 0 || wy >= chunk::CHUNK_H as i32 {
            return BlockType::Air;
        }
        let cx = wx.div_euclid(CHUNK_W as i32);
        let cz = wz.div_euclid(CHUNK_D as i32);
        let lx = wx.rem_euclid(CHUNK_W as i32) as usize;
        let lz = wz.rem_euclid(CHUNK_D as i32) as usize;
        self.chunks
            .get(&(cx, cz))
            .map(|c| *c.get(lx, wy as usize, lz))
            .unwrap_or(BlockType::Air)
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) {
        if wy < 0 || wy >= chunk::CHUNK_H as i32 {
            return;
        }
        let cx = wx.div_euclid(CHUNK_W as i32);
        let cz = wz.div_euclid(CHUNK_D as i32);
        let lx = wx.rem_euclid(CHUNK_W as i32) as usize;
        let lz = wz.rem_euclid(CHUNK_D as i32) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.set(lx, wy as usize, lz, block);
        }
    }

    /// Read block by world coords and return the numeric block ID
    pub fn get_block_id(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        self.get_block(wx, wy, wz).id()
    }

    /// Write block by world coords using numeric block ID
    pub fn set_block_id(&mut self, wx: i32, wy: i32, wz: i32, block_id: u8) {
        if let Some(block) = BlockType::from_id(block_id) {
            self.set_block(wx, wy, wz, block);
        }
    }

    /// Query ambient lighting color and multiplier at world coordinates.
    /// Returns `[r, g, b, multiplier]` where `multiplier` is applied to ambient term.
    pub fn ambient_at(&self, wx: i32, wz: i32) -> [f32; 4] {
        self.generator.ambient_at(wx, wz)
    }

    /// Return the surface height (y) at world coordinates using the biome generator.
    pub fn surface_height(&self, wx: i32, wz: i32) -> u32 {
        self.generator.surface_height(wx, wz)
    }

    /// Find a safe spawn position on land, spiralling outward from (0, 0) until a
    /// non-liquid surface is found.  Uses the biome generator directly so that no
    /// chunks need to be loaded at the time of the call.
    ///
    /// Returns `(x, y, z)` world coordinates where y already accounts for eye-height.
    pub fn find_safe_spawn(&self) -> (f32, f32, f32) {
        // Quick path: (0, 0) is already solid land
        if !self.generator.surface_is_liquid(0, 0) {
            let h = self.generator.surface_height(0, 0) as f32;
            return (0.5, h + 1.8, 0.5);
        }

        // Spiral outward in rings until we find a dry tile
        for radius in 1i32..=256 {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    // Only visit the outermost ring of this radius
                    if dx.abs() != radius && dz.abs() != radius {
                        continue;
                    }
                    if !self.generator.surface_is_liquid(dx, dz) {
                        let h = self.generator.surface_height(dx, dz) as f32;
                        return (dx as f32 + 0.5, h + 1.8, dz as f32 + 0.5);
                    }
                }
            }
        }

        // Absolute fallback: place well above any water
        let h = self.generator.surface_height(0, 0) as f32;
        (0.5, h.max(biomes::WATER_LEVEL as f32) + 1.8, 0.5)
    }

    /// Level-based water/liquid simulation.  Every water block is a liquid body
    /// with a level stored in bits 0-2 of its metadata (1 = furthest extent,
    /// 7 = adjacent to source or one step below).  Bit 3 marks a permanent
    /// source block that never drains.
    ///
    /// Flow rules (Minecraft-style):
    ///  - Downward:   always creates a full-level (7) flow block in the cell below.
    ///  - Lateral:    a source spreads at level 6; a flow block at level N spreads
    ///                at level N-1.  Spreading stops when level would reach 0.
    ///  - A lateral neighbour is updated only when our new level is strictly greater
    ///    than the neighbour's current level (no infinite-loop oscillation).
    ///
    /// Returns the set of chunk coordinates whose block data changed so that only
    /// those water meshes need to be rebuilt.
    pub fn simulate_water(&mut self) -> Vec<(i32, i32)> {
        let mut changes:      Vec<(i32, i32, i32, BlockType)> = Vec::new();
        let mut meta_changes: Vec<(i32, i32, i32, u8)>        = Vec::new();
        let mut dirty: HashSet<(i32, i32)>                    = HashSet::new();

        // Collect chunk keys upfront to avoid holding a borrow on self.chunks
        let chunk_keys: Vec<(i32, i32)> = self.chunks.keys().cloned().collect();

        for (cx, cz) in chunk_keys {
            let chunk = match self.chunks.get(&(cx, cz)) {
                Some(c) => c,
                None    => continue,
            };

            // Fast-skip chunks that contain no water blocks at all
            let has_water = chunk.blocks.iter()
                .flat_map(|yz| yz.iter())
                .flat_map(|z| z.iter())
                .any(|&b| b == BlockType::Water);
            if !has_water { continue; }

            for x in 0..CHUNK_W {
                for z in 0..CHUNK_D {
                    for y in 0..crate::world::chunk::CHUNK_H {
                        if *chunk.get(x, y, z) != BlockType::Water { continue; }

                        let wx = cx * CHUNK_W as i32 + x as i32;
                        let wy = y as i32;
                        let wz = cz * CHUNK_D as i32 + z as i32;

                        let meta      = self.get_water_meta(wx, wy, wz);
                        let is_source = (meta & 0x08) != 0;
                        let level     = meta & 0x07;

                        // ── Downward flow ─────────────────────────────────────────
                        if wy > 0 {
                            // Chunk coordinate of the cell below (may differ when y==0)
                            let below_cx = wx.div_euclid(CHUNK_W as i32);
                            let below_cz = wz.div_euclid(CHUNK_D as i32);
                            match self.get_block(wx, wy - 1, wz) {
                                BlockType::Air => {
                                    changes.push((wx, wy - 1, wz, BlockType::Water));
                                    meta_changes.push((wx, wy - 1, wz, 0x07)); // full level
                                    dirty.insert((below_cx, below_cz));
                                    if !is_source {
                                        changes.push((wx, wy, wz, BlockType::Air));
                                        meta_changes.push((wx, wy, wz, 0x00));
                                        dirty.insert((cx, cz));
                                    }
                                    continue; // downward takes priority over lateral
                                }
                                BlockType::Water => {
                                    // Equalise level below if we can raise it
                                    let below_meta = self.get_water_meta(wx, wy - 1, wz);
                                    if (below_meta & 0x08) == 0 && (below_meta & 0x07) < 7 {
                                        meta_changes.push((wx, wy - 1, wz, 0x07));
                                        dirty.insert((below_cx, below_cz));
                                    }
                                }
                                _ => {}
                            }
                        }

                        // ── Lateral spread ───────────────────────────────────────
                        // A source block spawns level-6 neighbours; a flow block
                        // spawns level (level-1) neighbours; stops at 0.
                        let spread_level: u8 = if is_source { 6 } else { level.saturating_sub(1) };
                        if spread_level == 0 { continue; }

                        for (dx, dz) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                            let nx = wx + dx;
                            let nz = wz + dz;
                            let ncx = nx.div_euclid(CHUNK_W as i32);
                            let ncz = nz.div_euclid(CHUNK_D as i32);

                            match self.get_block(nx, wy, nz) {
                                BlockType::Air => {
                                    // Require a solid or water floor so water doesn't
                                    // slide horizontally off unsupported cliffs
                                    let floor = if wy > 0 { self.get_block(nx, wy - 1, nz) } else { BlockType::Bedrock };
                                    if floor != BlockType::Air {
                                        changes.push((nx, wy, nz, BlockType::Water));
                                        meta_changes.push((nx, wy, nz, spread_level));
                                        dirty.insert((ncx, ncz));
                                        // Only consume a flow block (not a source) on lateral spread
                                        if !is_source {
                                            changes.push((wx, wy, wz, BlockType::Air));
                                            meta_changes.push((wx, wy, wz, 0x00));
                                            dirty.insert((cx, cz));
                                            break; // one direction per tick for flow blocks
                                        }
                                    }
                                }
                                BlockType::Water => {
                                    // Update neighbour's level only if ours is strictly higher
                                    let n_meta = self.get_water_meta(nx, wy, nz);
                                    let n_src  = (n_meta & 0x08) != 0;
                                    let n_lvl  = n_meta & 0x07;
                                    if !n_src && n_lvl < spread_level {
                                        meta_changes.push((nx, wy, nz, spread_level));
                                        dirty.insert((ncx, ncz));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        // Apply all changes
        for (wx, wy, wz, block) in changes {
            self.set_block(wx, wy, wz, block);
        }
        for (wx, wy, wz, meta) in meta_changes {
            self.set_water_meta(wx, wy, wz, meta);
        }

        dirty.into_iter().collect()
    }

    /// Get per-voxel water metadata (0 if none)
    pub fn get_water_meta(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        if wy < 0 || wy >= chunk::CHUNK_H as i32 { return 0; }
        let cx = wx.div_euclid(CHUNK_W as i32);
        let cz = wz.div_euclid(CHUNK_D as i32);
        let lx = wx.rem_euclid(CHUNK_W as i32) as usize;
        let lz = wz.rem_euclid(CHUNK_D as i32) as usize;
        self.chunks
            .get(&(cx, cz))
            .map(|c| c.water_meta_get(lx, wy as usize, lz))
            .unwrap_or(0)
    }

    /// Set per-voxel water metadata (no-op if chunk missing)
    pub fn set_water_meta(&mut self, wx: i32, wy: i32, wz: i32, meta: u8) {
        if wy < 0 || wy >= chunk::CHUNK_H as i32 { return; }
        let cx = wx.div_euclid(CHUNK_W as i32);
        let cz = wz.div_euclid(CHUNK_D as i32);
        let lx = wx.rem_euclid(CHUNK_W as i32) as usize;
        let lz = wz.rem_euclid(CHUNK_D as i32) as usize;
        if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
            chunk.water_meta_set(lx, wy as usize, lz, meta);
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create save directory")?;
        }

        let chunks = self.chunks.iter().map(|(&(cx, cz), chunk)| ChunkSave {
            cx,
            cz,
            blocks: chunk.flatten(),
        }).collect();

        let save = WorldSave {
            seed: self.generator.seed(),
            chunks,
        };

        let file = File::create(path).context("create save file")?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &save).context("serialize world save")?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).context("open save file")?;
        let save: WorldSave = serde_json::from_reader(file).context("deserialize world save")?;

        let mut world = World::new(save.seed);
        for chunk_save in save.chunks {
            let chunk = Chunk::from_flat(&chunk_save.blocks);
            world.chunks.insert((chunk_save.cx, chunk_save.cz), chunk);
        }

        Ok(world)
    }
}