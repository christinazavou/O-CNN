import os
import numpy as np
np.random.seed(100)


PARTNET_LABELS_LEVEL3 = {
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


COLORS = {}

for key, value in PARTNET_LABELS_LEVEL3.items():
    COLORS[key] = np.random.randint(0, 16777215, len(value))


def find_category(model_path):
    dirs = model_path.split(os.sep)
    categories = PARTNET_LABELS_LEVEL3.keys()
    for d in dirs:
        if d in categories:
            return d
    return None


def decimal_to_rgb(decimal):
    hexadecimal_str = '{:06x}'.format(decimal)
    return tuple(int(hexadecimal_str[i:i + 2], 16) for i in (0, 2, 4))


if __name__ == '__main__':

    for key, value in PARTNET_LABELS_LEVEL3.items():
        print(key, len(value))
        print("colors ", [decimal_to_rgb(col) for col in COLORS[key]])

    cat = find_category('/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/logs/seg/0811_partnet_randinit/Bottle/ratio_1.00/model/iter_000150.ckpt')
    print("category ", cat)
