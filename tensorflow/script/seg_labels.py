import os

import numpy as np

np.random.seed(100)


LEVEL3_FULL_LABELS = {
    'Bed': [
        'undefined',
        'bed/bed_unit/bed_sleep_area/pillow',
        'bed/bed_unit/bed_sleep_area/mattress',
        'bed/bed_unit/bed_frame/bed_frame_horizontal_surface/frame_horizontal_hard_surface',
        'bed/bed_unit/bed_frame/bed_frame_horizontal_surface/frame_horizontal_surface_bar',
        'bed/bed_unit/bed_frame/bed_frame_base/regular_leg_base/bar_stretcher',
        'bed/bed_unit/bed_frame/bed_frame_base/regular_leg_base/leg',
        'bed/bed_unit/bed_frame/bed_frame_base/surface_base',
        'bed/bed_unit/bed_frame/bed_side_surface/bed_side_surface_horizontal_bar',
        'bed/bed_unit/bed_frame/bed_side_surface/bed_side_surface_panel',
        'bed/bed_unit/bed_frame/bed_side_surface/bed_side_surface_vertical_bar',
        'bed/bed_unit/bed_frame/bed_post',
        'bed/bed_unit/bed_frame/headboard',
        'bed/ladder/ladder_vertical_bar',
        'bed/ladder/rung'
    ],
    'Bottle': [
        'undefined',
        'bottle/normal_bottle/body',
        'bottle/normal_bottle/closure',
        'bottle/normal_bottle/lid',
        'bottle/normal_bottle/handle',
        'bottle/normal_bottle/neck',
        'bottle/normal_bottle/mouth',
        'bottle/jug/body',
        'bottle/jug/handle'
    ],
    'Chair': [
        'undefined',
        'chair/chair_head/headrest',
        'chair/chair_head/head_connector',
        'chair/chair_back/back_surface/back_surface_vertical_bar',
        'chair/chair_back/back_surface/back_surface_horizontal_bar',
        'chair/chair_back/back_surface/back_single_surface',
        'chair/chair_back/back_connector',
        'chair/chair_back/back_support',
        'chair/chair_back/back_frame/back_frame_vertical_bar',
        'chair/chair_back/back_frame/back_frame_horizontal_bar',
        'chair/chair_back/back_frame/back_holistic_frame',
        'chair/chair_arm/arm_sofa_style',
        'chair/chair_arm/arm_horizontal_bar',
        'chair/chair_arm/arm_near_vertical_bar',
        'chair/chair_arm/arm_writing_table',
        'chair/chair_arm/arm_holistic_frame',
        'chair/chair_arm/arm_connector',
        'chair/chair_base/star_leg_base/star_leg_set/caster/wheel',
        'chair/chair_base/star_leg_base/star_leg_set/caster/caster_stem',
        'chair/chair_base/star_leg_base/star_leg_set/leg',
        'chair/chair_base/star_leg_base/central_support',
        'chair/chair_base/star_leg_base/mechanical_control/knob',
        'chair/chair_base/star_leg_base/mechanical_control/lever',
        'chair/chair_base/regular_leg_base/foot',
        'chair/chair_base/regular_leg_base/runner',
        'chair/chair_base/regular_leg_base/bar_stretcher',
        'chair/chair_base/regular_leg_base/rocker',
        'chair/chair_base/regular_leg_base/leg',
        'chair/chair_base/foot_base/foot',
        'chair/chair_base/pedestal_base/pedestal',
        'chair/chair_base/pedestal_base/central_support',
        'chair/chair_seat/seat_support',
        'chair/chair_seat/seat_frame/seat_holistic_frame',
        'chair/chair_seat/seat_frame/seat_frame_bar',
        'chair/chair_seat/seat_surface/seat_surface_bar',
        'chair/chair_seat/seat_surface/seat_single_surface',
        'chair/footrest/chair_base',
        'chair/footrest/chair_seat/seat_support',
        'chair/footrest/chair_seat/seat_surface'
    ],
    'Clock': [
        'undefined',
        'clock/table_clock/clock_body/frame',
        'clock/table_clock/clock_body/surface',
        'clock/table_clock/base/foot_base/foot',
        'clock/table_clock/base/surface_base/base_surface',
        'clock/pendulum_clock/pendulum_clock_base/foot_base/foot',
        'clock/pendulum_clock/pendulum_clock_base/surface_base/base_surface',
        'clock/pendulum_clock/pendulum_clock_frame/box',
        'clock/pendulum_clock/pendulum/pendulum_body',
        'clock/pendulum_clock/pendulum/chain',
        'clock/pendulum_clock/pendulum_clock_top'
    ],
    'Dishwasher': [
        'undefined',
        'dishwasher/body/frame',
        'dishwasher/body/door/door_frame',
        'dishwasher/body/door/handle',
        'dishwasher/base/foot_base/foot',
        'dishwasher/base/foot_base/surface',
        'dishwasher/base/surface_base/surface'
    ],
    'Display': [
        'undefined',
        'display/display_screen',
        'display/base/surface_base/base_support',
        'display/base/surface_base/surface'
    ],
    'Door': [
        'undefined',
        'door_set/outside_frame',
        'door_set/door/handle/fixed_part',
        'door_set/door/handle/movable_part',
        'door_set/door/door_body/surface_board'
    ],
    'Earphone': [
        'undefined',
        'earphone/earbud/earbud_unit/earbud_connector',
        'earphone/earbud/earbud_unit/earbud_frame',
        'earphone/earbud/earbud_unit/earbud_pad',
        'earphone/earbud/earbud_connector_wire',
        'earphone/headphone/head_band/top_band',
        'earphone/headphone/earcup_unit/earcup_pad',
        'earphone/headphone/earcup_unit/earcup_frame',
        'earphone/headphone/earcup_unit/earcup_connector',
        'earphone/headphone/connector_wire'
    ],
    'Faucet': [
        'undefined',
        'faucet/shower_faucet/switch',
        'faucet/shower_faucet/hose',
        'faucet/shower_faucet/spout/tube',
        'faucet/shower_faucet/frame/horizontal_support',
        'faucet/shower_faucet/frame/vertical_support',
        'faucet/shower_faucet/frame/surface_base',
        'faucet/normal_faucet/spout/tube',
        'faucet/normal_faucet/switch',
        'faucet/normal_faucet/frame/horizontal_support',
        'faucet/normal_faucet/frame/vertical_support',
        'faucet/normal_faucet/frame/surface_base'
    ],
    'Knife': [
        'undefined',
        'cutting_instrument/dagger/handle_side/guard',
        'cutting_instrument/dagger/handle_side/handle',
        'cutting_instrument/dagger/handle_side/butt',
        'cutting_instrument/dagger/blade_side/blade',
        'cutting_instrument/knife/handle_side/guard',
        'cutting_instrument/knife/handle_side/handle',
        'cutting_instrument/knife/handle_side/butt',
        'cutting_instrument/knife/blade_side/bolster',
        'cutting_instrument/knife/blade_side/blade'
    ],
    'Lamp': [
        'undefined',
        'lamp/ceiling_lamp/chandelier/lamp_unit_group/lamp_unit/lamp_arm',
        'lamp/ceiling_lamp/chandelier/lamp_unit_group/lamp_unit/lamp_head/lamp_cover/lamp_shade',
        'lamp/ceiling_lamp/chandelier/lamp_unit_group/lamp_unit/lamp_head/light_bulb',
        'lamp/ceiling_lamp/chandelier/lamp_body',
        'lamp/ceiling_lamp/chandelier/chain',
        'lamp/ceiling_lamp/chandelier/lamp_base/lamp_holistic_base/lamp_base_part',
        'lamp/ceiling_lamp/pendant_lamp/power_cord/cord',
        'lamp/ceiling_lamp/pendant_lamp/pendant_lamp_unit/chain',
        'lamp/ceiling_lamp/pendant_lamp/pendant_lamp_unit/lamp_head/lamp_cover/lamp_shade',
        'lamp/ceiling_lamp/pendant_lamp/pendant_lamp_unit/lamp_head/light_bulb',
        'lamp/ceiling_lamp/pendant_lamp/lamp_base/lamp_holistic_base/lamp_base_part',
        'lamp/table_or_floor_lamp/power_cord/cord',
        'lamp/table_or_floor_lamp/lamp_body/lamp_pole',
        'lamp/table_or_floor_lamp/lamp_body/lamp_body_solid',
        'lamp/table_or_floor_lamp/lamp_body/lamp_body_jointed/lamp_arm/lamp_arm_straight_bar',
        'lamp/table_or_floor_lamp/lamp_body/lamp_body_vertical_panel',
        'lamp/table_or_floor_lamp/lamp_unit/connector',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_arm/lamp_arm_straight_bar',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_arm/lamp_arm_curved_bar',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_head/lamp_cover/lamp_shade',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_head/light_bulb',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_head/lamp_finial',
        'lamp/table_or_floor_lamp/lamp_unit/lamp_head/lamp_wireframe_fitter',
        'lamp/table_or_floor_lamp/lamp_base/lamp_holistic_base/lamp_base_part',
        'lamp/table_or_floor_lamp/lamp_base/lamp_leg_base/leg',
        'lamp/wall_lamp/lamp_body',
        'lamp/wall_lamp/lamp_unit/lamp_arm/lamp_arm_straight_bar',
        'lamp/wall_lamp/lamp_unit/lamp_arm/lamp_arm_curved_bar',
        'lamp/wall_lamp/lamp_unit/lamp_head/lamp_cover/lamp_shade',
        'lamp/wall_lamp/lamp_base/lamp_holistic_base/lamp_base_part',
        'lamp/street_lamp/lamp_post',
        'lamp/street_lamp/street_lamp_base',
        'lamp/street_lamp/lamp_unit/lamp_arm/lamp_arm_straight_bar',
        'lamp/street_lamp/lamp_unit/lamp_arm/lamp_arm_curved_bar',
        'lamp/street_lamp/lamp_unit/lamp_head/lamp_cover/lamp_shade',
        'lamp/street_lamp/lamp_unit/lamp_head/lamp_cover/lantern_lamp_cover/lamp_cover_frame/lamp_cover_frame_top',
        'lamp/street_lamp/lamp_unit/lamp_head/lamp_cover/lantern_lamp_cover/lamp_cover_frame/lamp_cover_frame_bottom',
        'lamp/street_lamp/lamp_unit/lamp_head/lamp_cover/lantern_lamp_cover/lamp_cover_frame/lamp_cover_frame_bar',
        'lamp/street_lamp/lamp_unit/lamp_head/light_bulb',
        'lamp/street_lamp/lamp_unit/lamp_head/lamp_cover_holder'
    ],
    'Microwave': [
        'undefined',
        'microwave/body/frame',
        'microwave/body/door/door_frame',
        'microwave/body/door/handle',
        'microwave/body/body_interior/tray',
        'microwave/base/foot_base/foot'
    ],
    'Refrigerator': [
        'undefined',
        'refrigerator/body/frame',
        'refrigerator/body/door/door_frame',
        'refrigerator/body/door/handle',
        'refrigerator/body/body_interior/shelf',
        'refrigerator/base/foot_base/foot',
        'refrigerator/base/surface_base/surface'
    ],
    'StorageFurniture': [
        'undefined',
        'storage_furniture/cabinet/countertop',
        'storage_furniture/cabinet/shelf',
        'storage_furniture/cabinet/cabinet_frame/frame_vertical_bar',
        'storage_furniture/cabinet/cabinet_frame/back_panel',
        'storage_furniture/cabinet/cabinet_frame/top_panel',
        'storage_furniture/cabinet/cabinet_frame/vertical_side_panel',
        'storage_furniture/cabinet/cabinet_frame/frame_horizontal_bar',
        'storage_furniture/cabinet/cabinet_frame/vertical_front_panel',
        'storage_furniture/cabinet/cabinet_frame/bottom_panel',
        'storage_furniture/cabinet/cabinet_frame/vertical_divider_panel',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_back',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_bottom',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_side',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_front',
        'storage_furniture/cabinet/drawer/handle',
        'storage_furniture/cabinet/cabinet_base/panel_base/bottom_panel',
        'storage_furniture/cabinet/cabinet_base/panel_base/base_side_panel',
        'storage_furniture/cabinet/cabinet_base/foot_base/foot',
        'storage_furniture/cabinet/cabinet_base/foot_base/caster/wheel',
        'storage_furniture/cabinet/cabinet_base/foot_base/caster/caster_stem',
        'storage_furniture/cabinet/cabinet_door/hinge',
        'storage_furniture/cabinet/cabinet_door/handle',
        'storage_furniture/cabinet/cabinet_door/cabinet_door_surface'
    ],
    'Table': [
        'undefined',
        'table/game_table/ping_pong_table/ping_pong_net',
        'table/game_table/ping_pong_table/tabletop/tabletop_surface',
        'table/game_table/ping_pong_table/table_base/regular_leg_base/bar_stretcher',
        'table/game_table/ping_pong_table/table_base/regular_leg_base/leg',
        'table/game_table/pool_table/pool_ball',
        'table/game_table/pool_table/tabletop/tabletop_surface',
        'table/game_table/pool_table/tabletop/tabletop_frame/bar',
        'table/game_table/pool_table/table_base/regular_leg_base/leg',
        'table/picnic_table/regular_table/tabletop/tabletop_surface',
        'table/picnic_table/regular_table/table_base/regular_leg_base/leg',
        'table/picnic_table/bench_connector',
        'table/picnic_table/bench',
        'table/regular_table/tabletop/tabletop_surface/glass',
        'table/regular_table/tabletop/tabletop_surface/bar',
        'table/regular_table/tabletop/tabletop_surface/board',
        'table/regular_table/tabletop/tabletop_dropleaf',
        'table/regular_table/tabletop/tabletop_frame/bar',
        'table/regular_table/table_base/star_leg_base/star_leg_set/leg',
        'table/regular_table/table_base/star_leg_base/central_support',
        'table/regular_table/table_base/regular_leg_base/tabletop_connector',
        'table/regular_table/table_base/regular_leg_base/bar_stretcher',
        'table/regular_table/table_base/regular_leg_base/leg',
        'table/regular_table/table_base/regular_leg_base/runner',
        'table/regular_table/table_base/regular_leg_base/circular_stretcher',
        'table/regular_table/table_base/regular_leg_base/foot',
        'table/regular_table/table_base/regular_leg_base/caster/wheel',
        'table/regular_table/table_base/regular_leg_base/caster/caster_stem',
        'table/regular_table/table_base/drawer_base/tabletop_connector',
        'table/regular_table/table_base/drawer_base/back_panel',
        'table/regular_table/table_base/drawer_base/bar_stretcher',
        'table/regular_table/table_base/drawer_base/leg',
        'table/regular_table/table_base/drawer_base/vertical_side_panel',
        'table/regular_table/table_base/drawer_base/shelf',
        'table/regular_table/table_base/drawer_base/cabinet_door/handle',
        'table/regular_table/table_base/drawer_base/cabinet_door/cabinet_door_surface',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_back',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_bottom',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_side',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_front',
        'table/regular_table/table_base/drawer_base/drawer/handle',
        'table/regular_table/table_base/drawer_base/keyboard_tray/keyboard_tray_surface',
        'table/regular_table/table_base/drawer_base/vertical_front_panel',
        'table/regular_table/table_base/drawer_base/foot',
        'table/regular_table/table_base/drawer_base/bottom_panel',
        'table/regular_table/table_base/drawer_base/caster/wheel',
        'table/regular_table/table_base/drawer_base/caster/caster_stem',
        'table/regular_table/table_base/drawer_base/vertical_divider_panel',
        'table/regular_table/table_base/pedestal_base/pedestal',
        'table/regular_table/table_base/pedestal_base/tabletop_connector',
        'table/regular_table/table_base/pedestal_base/central_support'
    ],
    'TrashCan': [
        'undefined',
        'trash_can/container/container_bottom',
        'trash_can/container/container_box',
        'trash_can/container/container_neck',
        'trash_can/outside_frame/frame_vertical_bar',
        'trash_can/outside_frame/frame_horizontal_circle',
        'trash_can/outside_frame/frame_bottom',
        'trash_can/outside_frame/frame_holistic',
        'trash_can/base/foot',
        'trash_can/cover/cover_support',
        'trash_can/cover/cover_lid'
    ],
    'Vase': [
        'undefined',
        'pot/body/lid',
        'pot/body/container',
        'pot/base/foot_base/foot',
        'pot/containing_things/plant',
        'pot/containing_things/liquid_or_soil'
    ]
}


LEVEL3_LABELS = {
    'Bed': [
        'undefined',
        'pillow',
        'mattress',
        'frame_horizontal_hard_surface',
        'frame_horizontal_surface_bar',
        'bar_stretcher',
        'leg',
        'surface_base',
        'bed_side_surface_horizontal_bar',
        'bed_side_surface_panel',
        'bed_side_surface_vertical_bar',
        'bed_post',
        'headboard',
        'ladder_vertical_bar',
        'rung'
    ],
    'Bottle': [
        'undefined',
        'normal_bottle_body',
        'closure',
        'lid',
        'normal_bottle_handle',
        'neck',
        'mouth',
        'jug_body',
        'jug_handle'
    ],
    'Chair': [
        'undefined',
        'headrest',
        'head_connector',
        'back_surface_vertical_bar',
        'back_surface_horizontal_bar',
        'back_single_surface',
        'back_connector',
        'back_support',
        'back_frame_vertical_bar',
        'back_frame_horizontal_bar',
        'back_holistic_frame',
        'arm_sofa_style',
        'arm_horizontal_bar',
        'arm_near_vertical_bar',
        'arm_writing_table',
        'arm_holistic_frame',
        'arm_connector',
        'wheel',
        'caster_stem',
        'star_leg_set_leg',
        'star_leg_base_central_support',
        'knob',
        'lever',
        'regular_leg_base_foot',
        'runner',
        'bar_stretcher',
        'rocker',
        'regular_leg_base_leg',
        'foot_base_foot',
        'pedestal',
        'pedestal_base_central_support',
        'chair_seat_seat_support',
        'seat_holistic_frame',
        'seat_frame_bar',
        'seat_surface_bar',
        'seat_single_surface',
        'chair_base',
        'footrest_seat_support',
        'footrest_seat_surface'
    ],
    'Clock': [
        'undefined',
        'frame',
        'surface',
        'table_clock_foot',
        'table_clock_base_surface',
        'pendulum_foot',
        'pendulum_base_surface',
        'box',
        'pendulum_body',
        'chain',
        'pendulum_clock_top'
    ],
    'Dishwasher': [
        'undefined',
        'frame',
        'door_frame',
        'handle',
        'foot',
        'foot_base_surface',
        'surface_base_surface'
    ],
    'Display': [
        'undefined',
        'display_screen',
        'base_support',
        'surface'
    ],
    'Door': [
        'undefined',
        'outside_frame',
        'fixed_part',
        'movable_part',
        'surface_board'
    ],
    'Earphone': [
        'undefined',
        'earbud_connector',
        'earbud_frame',
        'earbud_pad',
        'earbud_connector_wire',
        'top_band',
        'earcup_pad',
        'earcup_frame',
        'earcup_connector',
        'headphone_connector_wire'
    ],
    'Faucet': [
        'undefined',
        'shower_fct_switch',
        'shower_fct_hose',
        'shower_fct_tube',
        'shower_fct_horiz_support',
        'shower_fct_vert_support',
        'shower_fct_surface_base',
        'normal_fct_tube',
        'normal_fct_switch',
        'normal_fct_horiz_support',
        'normal_fct_vert_support',
        'normal_fct_surface_base'
    ],
    'Knife': [
        'undefined',
        'dagger_handle_guard',
        'dagger_handle',
        'dagger_handle_butt',
        'dagger_blade',
        'knife_handle_guard',
        'knife_handle',
        'knife_handle_butt',
        'knife_blade_bolster',
        'knife_blade'
    ],
    'Lamp': [
        'undefined',
        'ceiling_chandelier_arm',
        'ceiling_chandelier_shade',
        'ceiling_chandelier_light_bulb',
        'ceiling_chandelier_body',
        'ceiling_chandelier_chain',
        'ceiling_chandelier_base_part',
        'ceiling_pendant_cord',
        'ceiling_pendant_chain',
        'ceiling_pendant_shade',
        'ceiling_pendant_light_bulb',
        'ceiling_pendant_base_part',
        'floor_cord',
        'floor_pole',
        'floor_body_solid',
        'floor_body_joint_arm_straight_bar',
        'floor_body_vertical_panel',
        'floor_connector',
        'floor_arm_straight_bar',
        'floor_arm_curved_bar',
        'floor_shade',
        'floor_light_bulb',
        'floor_finial',
        'floor_wireframe_fitter',
        'floor_base_part',
        'floor_leg',
        'wall_body',
        'wall_arm_straight_bar',
        'wall_arm_curved_bar',
        'wall_shade',
        'wall_lamp_base_part',
        'street_post',
        'street_base',
        'street_arm_straight_bar',
        'street_arm_curved_bar',
        'street_shade',
        'street_cover_frame_top',
        'street_cover_frame_bottom',
        'street_cover_frame_bar',
        'street_light_bulb',
        'street_cover_holder'
    ],
    'Microwave': [
        'undefined',
        'frame',
        'door_frame',
        'handle',
        'tray',
        'foot'
    ],
    'Refrigerator': [
        'undefined',
        'frame',
        'door_frame',
        'handle',
        'shelf',
        'foot',
        'surface'
    ],
'StorageFurniture': [
        'undefined',
        'storage_furniture/cabinet/countertop',
        'storage_furniture/cabinet/shelf',
        'storage_furniture/cabinet/cabinet_frame/frame_vertical_bar',
        'storage_furniture/cabinet/cabinet_frame/back_panel',
        'storage_furniture/cabinet/cabinet_frame/top_panel',
        'storage_furniture/cabinet/cabinet_frame/vertical_side_panel',
        'storage_furniture/cabinet/cabinet_frame/frame_horizontal_bar',
        'storage_furniture/cabinet/cabinet_frame/vertical_front_panel',
        'storage_furniture/cabinet/cabinet_frame/bottom_panel',
        'storage_furniture/cabinet/cabinet_frame/vertical_divider_panel',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_back',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_bottom',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_side',
        'storage_furniture/cabinet/drawer/drawer_box/drawer_front',
        'storage_furniture/cabinet/drawer/handle',
        'storage_furniture/cabinet/cabinet_base/panel_base/bottom_panel',
        'storage_furniture/cabinet/cabinet_base/panel_base/base_side_panel',
        'storage_furniture/cabinet/cabinet_base/foot_base/foot',
        'storage_furniture/cabinet/cabinet_base/foot_base/caster/wheel',
        'storage_furniture/cabinet/cabinet_base/foot_base/caster/caster_stem',
        'storage_furniture/cabinet/cabinet_door/hinge',
        'storage_furniture/cabinet/cabinet_door/handle',
        'storage_furniture/cabinet/cabinet_door/cabinet_door_surface'
    ],
    'Table': [
        'undefined',
        'table/game_table/ping_pong_table/ping_pong_net',
        'table/game_table/ping_pong_table/tabletop/tabletop_surface',
        'table/game_table/ping_pong_table/table_base/regular_leg_base/bar_stretcher',
        'table/game_table/ping_pong_table/table_base/regular_leg_base/leg',
        'table/game_table/pool_table/pool_ball',
        'table/game_table/pool_table/tabletop/tabletop_surface',
        'table/game_table/pool_table/tabletop/tabletop_frame/bar',
        'table/game_table/pool_table/table_base/regular_leg_base/leg',
        'table/picnic_table/regular_table/tabletop/tabletop_surface',
        'table/picnic_table/regular_table/table_base/regular_leg_base/leg',
        'table/picnic_table/bench_connector',
        'table/picnic_table/bench',
        'table/regular_table/tabletop/tabletop_surface/glass',
        'table/regular_table/tabletop/tabletop_surface/bar',
        'table/regular_table/tabletop/tabletop_surface/board',
        'table/regular_table/tabletop/tabletop_dropleaf',
        'table/regular_table/tabletop/tabletop_frame/bar',
        'table/regular_table/table_base/star_leg_base/star_leg_set/leg',
        'table/regular_table/table_base/star_leg_base/central_support',
        'table/regular_table/table_base/regular_leg_base/tabletop_connector',
        'table/regular_table/table_base/regular_leg_base/bar_stretcher',
        'table/regular_table/table_base/regular_leg_base/leg',
        'table/regular_table/table_base/regular_leg_base/runner',
        'table/regular_table/table_base/regular_leg_base/circular_stretcher',
        'table/regular_table/table_base/regular_leg_base/foot',
        'table/regular_table/table_base/regular_leg_base/caster/wheel',
        'table/regular_table/table_base/regular_leg_base/caster/caster_stem',
        'table/regular_table/table_base/drawer_base/tabletop_connector',
        'table/regular_table/table_base/drawer_base/back_panel',
        'table/regular_table/table_base/drawer_base/bar_stretcher',
        'table/regular_table/table_base/drawer_base/leg',
        'table/regular_table/table_base/drawer_base/vertical_side_panel',
        'table/regular_table/table_base/drawer_base/shelf',
        'table/regular_table/table_base/drawer_base/cabinet_door/handle',
        'table/regular_table/table_base/drawer_base/cabinet_door/cabinet_door_surface',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_back',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_bottom',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_side',
        'table/regular_table/table_base/drawer_base/drawer/drawer_box/drawer_front',
        'table/regular_table/table_base/drawer_base/drawer/handle',
        'table/regular_table/table_base/drawer_base/keyboard_tray/keyboard_tray_surface',
        'table/regular_table/table_base/drawer_base/vertical_front_panel',
        'table/regular_table/table_base/drawer_base/foot',
        'table/regular_table/table_base/drawer_base/bottom_panel',
        'table/regular_table/table_base/drawer_base/caster/wheel',
        'table/regular_table/table_base/drawer_base/caster/caster_stem',
        'table/regular_table/table_base/drawer_base/vertical_divider_panel',
        'table/regular_table/table_base/pedestal_base/pedestal',
        'table/regular_table/table_base/pedestal_base/tabletop_connector',
        'table/regular_table/table_base/pedestal_base/central_support'
    ],
    'TrashCan': [
        'undefined',
        'trash_can/container/container_bottom',
        'trash_can/container/container_box',
        'trash_can/container/container_neck',
        'trash_can/outside_frame/frame_vertical_bar',
        'trash_can/outside_frame/frame_horizontal_circle',
        'trash_can/outside_frame/frame_bottom',
        'trash_can/outside_frame/frame_holistic',
        'trash_can/base/foot',
        'trash_can/cover/cover_support',
        'trash_can/cover/cover_lid'
    ],
    'Vase': [
        'undefined',
        'pot/body/lid',
        'pot/body/container',
        'pot/base/foot_base/foot',
        'pot/containing_things/plant',
        'pot/containing_things/liquid_or_soil'
    ]
}

ANNFASS_LABELS = {
    "Building": [
        # "empty",
        "undetermined",
        "wall",
        "window",
        "vehicle",
        "roof",
        "plant_tree",
        "door",
        "tower_steeple",
        "furniture",
        "ground_grass",
        "beam_frame",
        "stairs",
        "column",
        "railing_baluster",
        "floor",
        "chimney",
        "ceiling",
        "fence",
        "pond_pool",
        "corridor_path",
        "balcony_patio",
        "garage",
        "dome",
        "road",
        "entrance_gate",
        "parapet_merlon",
        "buttress",
        "dormer",
        "lantern_lamp",
        "arch",
        "awning",
        "shutters",
        "ramp",
        "canopy_gazebo"
    ]
}

LEVEL3_COLORS = {}
for key, value in LEVEL3_LABELS.items():
    LEVEL3_COLORS[key] = np.random.randint(0, 16777215, len(value))
    LEVEL3_COLORS[key][0] = 8355711  # grey for 'undefined'

ANNFASS_COLORS = {
    "Building": [
        # "#000000",  # -1
        "#000000",  # 0
        "#ff4500",  # 1
        "#0000ff",  # 2
        "#396073",  # 3
        "#4b008c",  # 4
        "#fa8072",  # 5
        "#7f0000",  # 6
        "#d6f2b6",  # 7
        "#0d2133",  # 8
        "#204035",  # 9
        "#ff4040",  # 10
        "#60b9bf",  # 11
        "#3d4010",  # 12
        "#733d00",  # 13
        "#400000",  # 14
        "#999673",  # 15
        "#ff00ff",  # 16
        "#394173",  # 17
        "#553df2",  # 18
        "#bf3069",  # 19
        "#301040",  # 20
        "#ff9180",  # 21
        "#997391",  # 22
        "#ffbfd9",  # 23
        "#00aaff",  # 24
        "#8a4d99",  # 25
        "#40ff73",  # 26
        "#8c6e69",  # 27
        "#cc00ff",  # 28
        "#b24700",  # 29
        "#ffbbdd",  # 30
        "#0dd3ff",  # 31
        "#00401a",  # 32
        "#c3e639",  # 33
    ]
}


def find_category(model_path, categories):
    dirs = model_path.split(os.sep)
    categories = categories.keys()
    for d in dirs:
        if d in categories:
            return d
    return None


def decimal_to_rgb(decimal):
    hexadecimal_str = '{:06x}'.format(decimal)
    return tuple(int(hexadecimal_str[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb(hex):
    return tuple(int(hex[1:][i:i + 2], 16) for i in (0, 2, 4))


def to_rgb(value):
    if isinstance(value, str) and '#' in value:
        return hex_to_rgb(value)
    assert isinstance(value, np.int64) , " given value {} is of type {}".format(value, type(value))
    return decimal_to_rgb(value)


def get_level3_category_labels(cat):
    return [l.split("/")[-1] for l in LEVEL3_LABELS[cat]]


if __name__ == '__main__':

    for key, value in LEVEL3_LABELS.items():
        print(key, len(value))
        print("colors ", [to_rgb(col) for col in LEVEL3_COLORS[key]])

    cat = find_category('/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/logs/seg/0811_partnet_randinit/Bottle/ratio_1.00/model/iter_000150.ckpt', LEVEL3_LABELS)
    print("category ", cat)