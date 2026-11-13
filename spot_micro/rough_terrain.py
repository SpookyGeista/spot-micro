"""Isaac terrain generation plus local height queries used only by the simulator."""
import numpy as np
import torch
from isaaclab.terrains import (
    TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg, MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg, HfWaveTerrainCfg, HfPyramidSlopedTerrainCfg,
)
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


class HeightQueryTerrain(TerrainImporter):
    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        # The current TerrainImporter no longer retains its source mesh.
        self.height_query_mesh = convert_to_warp_mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), device=self.device)

    def ground_heights(self, points):
        starts = points.clone()
        starts[..., 2] += 2.
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1
        hits, _, _, _ = raycast_mesh(starts, directions, self.height_query_mesh, max_dist=10.)
        return hits[..., 2]


def configure_rough_terrain(cfg, settings, seed):
    size = tuple(settings.get("patch_size_m", [8., 8.]))
    generator = TerrainGeneratorCfg(
        seed=seed, size=size, num_rows=settings.get("levels", 6), num_cols=8,
        curriculum=True, border_width=3., horizontal_scale=.05, vertical_scale=.001,
        slope_threshold=.75, use_cache=False, color_scheme="height",
        sub_terrains={
            "flat": MeshPlaneTerrainCfg(proportion=.25),
            "waves": HfWaveTerrainCfg(proportion=.25, amplitude_range=tuple(settings.get("wave_amplitude_m", [.003, .025])), num_waves=10),
            "blocks": MeshRandomGridTerrainCfg(proportion=.25, grid_width=.26,
                grid_height_range=tuple(settings.get("block_height_m", [.003, .035])), platform_width=.6),
            "slopes": HfPyramidSlopedTerrainCfg(proportion=.25, slope_range=tuple(settings.get("slope_range", [.01, .18])), platform_width=.6),
        })
    cfg.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=generator,
        class_type=HeightQueryTerrain, max_init_terrain_level=settings.get("initial_level", 0),
        collision_group=-1, physics_material=cfg.terrain.physics_material, debug_vis=False)
