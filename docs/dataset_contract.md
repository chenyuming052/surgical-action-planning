# Dataset Contract

## registry.json
每条记录对应一个 physical recording / canonical video。

必备字段：
- canonical_id
- split
- coverage_group
- in_cholec80 / in_cholect50 / in_endoscapes
- has_cholec80_cvs / has_endoscapes_cvs
- cholec80_tool_presence
- endoscapes_public_id
- labels_available
- frame_counts
- file_paths

## cholect50 npz
- frames: int32 [T]
- triplets: uint8 [T,100]
- instruments: uint8 [T,6]
- verbs: uint8 [T,10]
- targets: uint8 [T,15]
- phase: int64 [T]
- triplet_groups: uint8 [T,G]
- ttc_target_inst: int16 [T]
- ttc_target_group: int16 [T]
- is_change_inst: bool [T]
- is_change_group: bool [T]
- is_censored_inst: bool [T]
- is_censored_group: bool [T]
- is_clipping: bool [T]

## cholec80 npz
- frames: int32 [T]
- phase_ids: int64 [T]
- tool_presence: uint8 [T,7]
- instrument_mapped: uint8 [T,6]

## cholec80_cvs npz
- frames: int32 [T]
- cvs_c1: int8 [T]
- cvs_c2: int8 [T]
- cvs_c3: int8 [T]
- cvs_score: int8 [T]
- has_cvs_label: bool [T]

## endoscapes npz
- frames: int32 [T]
- cvs_scores: float32 [T,3]
- in_roi: bool [T]
- anatomy_presence: uint8 [T,5]
- anatomy_obs_mask: bool [T,5]
