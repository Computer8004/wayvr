use std::{
    array::from_fn,
    mem::transmute,
    time::{Duration, Instant},
};

use glam::{Affine3A, Mat3A, Quat, Vec3, Vec3A, bool};
use libmonado::{self as mnd, DeviceLogic};
use openxr::{self as xr, Quaternionf, Vector2f, Vector3f};
use serde::{Deserialize, Serialize};
use wlx_common::{config::HandsfreePointer, config_io, overlays::ToastTopic};

use crate::{
    backend::input::{Haptics, InputState, Pointer, TrackedDevice, TrackedDeviceRole},
    overlays::toast::Toast,
    state::{AppSession, AppState},
};

use super::{XrState, helpers::posef_to_transform};

static CLICK_TIMES: [Duration; 3] = [
    Duration::ZERO,
    Duration::from_millis(500),
    Duration::from_millis(750),
];

pub(super) struct OpenXrInputSource {
    action_set: xr::ActionSet,
    pointers: [OpenXrPointer; 2],
    handsfree_pointer: OpenXrPointer,
    hand_tracking: Option<OpenXrHandTracking>,
}

pub(super) struct OpenXrPointer {
    source: OpenXrHandSource,
    space: xr::Space,
}

struct OpenXrHandTracking {
    trackers: [xr::HandTracker; 2],
    logged_joint_success: [bool; 2],
    last_missing_joint_log: [Instant; 2],
    last_joint_error_log: [Instant; 2],
    interaction_enabled: bool,
    palms_up_since: Option<Instant>,
    last_toggle: Instant,
    last_toggle_progress_stage: u8,
}

#[derive(Clone, Copy)]
struct DerivedHandInput {
    pose: Affine3A,
    pinch_strength: f32,
    grab_strength: f32,
    palm_up: bool,
}

pub struct MultiClickHandler<const COUNT: usize> {
    name: String,
    action_f32: xr::Action<f32>,
    action_bool: xr::Action<bool>,
    previous: [Instant; COUNT],
    held_active: bool,
    held_inactive: bool,
}

impl<const COUNT: usize> MultiClickHandler<COUNT> {
    fn new(action_set: &xr::ActionSet, action_name: &str, side: &str) -> anyhow::Result<Self> {
        let name = format!("{side}_{COUNT}-{action_name}");
        let name_f32 = format!("{}_value", &name);

        let action_bool = action_set.create_action::<bool>(&name, &name, &[])?;
        let action_f32 = action_set.create_action::<f32>(&name_f32, &name_f32, &[])?;

        Ok(Self {
            name,
            action_f32,
            action_bool,
            previous: from_fn(|_| Instant::now()),
            held_active: false,
            held_inactive: false,
        })
    }
    fn check<G>(&mut self, session: &xr::Session<G>, threshold: f32) -> anyhow::Result<bool> {
        let res = self.action_bool.state(session, xr::Path::NULL)?;
        let mut state = res.is_active && res.current_state;

        if !state {
            let res = self.action_f32.state(session, xr::Path::NULL)?;
            state = res.is_active && res.current_state >= threshold - 0.001;
        }

        if !state {
            self.held_active = false;
            self.held_inactive = false;
            return Ok(false);
        }

        if self.held_active {
            return Ok(true);
        }

        if self.held_inactive {
            return Ok(false);
        }

        let passed = self
            .previous
            .iter()
            .all(|instant| instant.elapsed() < CLICK_TIMES[COUNT]);

        if passed {
            log::trace!("{}: passed", self.name);
            self.held_active = true;
            self.held_inactive = false;

            // reset to no prior clicks
            let long_ago = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
            self.previous
                .iter_mut()
                .for_each(|instant| *instant = long_ago);
        } else if COUNT > 0 {
            log::trace!("{}: rotate", self.name);
            self.previous.rotate_right(1);
            self.previous[0] = Instant::now();
            self.held_inactive = true;
        }

        Ok(passed)
    }
}

pub struct CustomClickAction {
    single: MultiClickHandler<0>,
    double: MultiClickHandler<1>,
    triple: MultiClickHandler<2>,
}

impl CustomClickAction {
    pub fn new(action_set: &xr::ActionSet, name: &str, side: &str) -> anyhow::Result<Self> {
        let single = MultiClickHandler::new(action_set, name, side)?;
        let double = MultiClickHandler::new(action_set, name, side)?;
        let triple = MultiClickHandler::new(action_set, name, side)?;

        Ok(Self {
            single,
            double,
            triple,
        })
    }
    pub fn state(
        &mut self,
        before: bool,
        state: &XrState,
        session: &AppSession,
    ) -> anyhow::Result<bool> {
        let threshold = if before {
            session.config.xr_click_sensitivity_release
        } else {
            session.config.xr_click_sensitivity
        };

        Ok(self.single.check(&state.session, threshold)?
            || self.double.check(&state.session, threshold)?
            || self.triple.check(&state.session, threshold)?)
    }
}

pub(super) struct OpenXrHandSource {
    pose: xr::Action<xr::Posef>,
    click: CustomClickAction,
    grab: CustomClickAction,
    alt_click: CustomClickAction,
    show_hide: CustomClickAction,
    toggle_dashboard: CustomClickAction,
    space_drag: CustomClickAction,
    space_rotate: CustomClickAction,
    space_reset: CustomClickAction,
    modifier_right: CustomClickAction,
    modifier_middle: CustomClickAction,
    move_mouse: CustomClickAction,
    scroll: xr::Action<Vector2f>,
    haptics: xr::Action<xr::Haptic>,
}

impl OpenXrInputSource {
    pub fn new(xr: &XrState) -> anyhow::Result<Self> {
        let mut action_set =
            xr.session
                .instance()
                .create_action_set("wayvr", "WayVR Actions", 0)?;

        let left_source = OpenXrHandSource::new(&mut action_set, "left")?;
        let right_source = OpenXrHandSource::new(&mut action_set, "right")?;
        let fallback_source = OpenXrHandSource::new(&mut action_set, "handsfree")?;

        suggest_bindings(
            &xr.instance,
            &[&left_source, &right_source, &fallback_source],
        );

        xr.session.attach_action_sets(&[&action_set])?;

        Ok(Self {
            action_set,
            pointers: [
                OpenXrPointer::new(xr, left_source)?,
                OpenXrPointer::new(xr, right_source)?,
            ],
            handsfree_pointer: OpenXrPointer::new(xr, fallback_source)?,
            hand_tracking: OpenXrHandTracking::new(xr),
        })
    }

    pub fn haptics(&self, xr: &XrState, hand: usize, haptics: &Haptics) {
        let action = &self.pointers[hand].source.haptics;

        let duration_nanos = f64::from(haptics.duration) * 1_000_000_000.0;

        let _ = action.apply_feedback(
            &xr.session,
            xr::Path::NULL,
            &xr::HapticVibration::new()
                .amplitude(haptics.intensity)
                .frequency(haptics.frequency)
                .duration(xr::Duration::from_nanos(duration_nanos as _)),
        );
    }

    pub fn update(&mut self, xr: &XrState, state: &mut AppState) -> anyhow::Result<()> {
        xr.session.sync_actions(&[(&self.action_set).into()])?;

        let loc = xr.view.locate(&xr.stage, xr.predicted_display_time)?;
        let hmd = posef_to_transform(&loc.pose);
        let mut hmd_tracked = true;
        if loc
            .location_flags
            .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        {
            state.input_state.hmd.matrix3 = hmd.matrix3;
        } else {
            hmd_tracked = false;
        }

        if loc
            .location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID)
        {
            state.input_state.hmd.translation = hmd.translation;
        } else {
            hmd_tracked = false;
        }

        let mut any_tracked = false;
        for i in 0..2 {
            let pointer = &mut state.input_state.pointers[i];
            self.pointers[i].update(pointer, xr, &state.session)?;
            any_tracked |= pointer.tracked;
        }

        if let Some(hand_tracking) = self.hand_tracking.as_mut() {
            any_tracked |= hand_tracking.update(state, xr);
        }

        if !any_tracked {
            self.handsfree_pointer.update_handsfree(
                &mut state.input_state.pointers[0],
                xr,
                &state.session,
                hmd,
                hmd_tracked,
            )?;
        }

        Ok(())
    }

    fn update_device_battery_status(
        device: &mut mnd::Device,
        role: TrackedDeviceRole,
        input_state: &mut InputState,
    ) {
        if let Ok(status) = device.battery_status()
            && status.present
        {
            input_state.devices.push(TrackedDevice {
                soc: Some(status.charge),
                charging: status.charging,
                role,
            });
            log::debug!(
                "Device {} role {:#?}: {:.0}% (charging {})",
                device.index,
                role,
                status.charge * 100.0f32,
                status.charging
            );
        }
    }

    pub fn update_devices(app: &mut AppState) -> bool {
        let Some(monado) = &mut app.monado_state else {
            return false; // monado not available
        };

        let old_len = app.input_state.devices.len();
        app.input_state.devices.clear();

        let roles = [
            (mnd::DeviceRole::Head, TrackedDeviceRole::Hmd),
            (mnd::DeviceRole::Eyes, TrackedDeviceRole::None),
            (mnd::DeviceRole::Left, TrackedDeviceRole::LeftHand),
            (mnd::DeviceRole::Right, TrackedDeviceRole::RightHand),
            (mnd::DeviceRole::Gamepad, TrackedDeviceRole::None),
            (
                mnd::DeviceRole::HandTrackingLeft,
                TrackedDeviceRole::LeftHand,
            ),
            (
                mnd::DeviceRole::HandTrackingRight,
                TrackedDeviceRole::RightHand,
            ),
        ];
        let mut seen = Vec::<u32>::with_capacity(32);

        for (mnd_role, wlx_role) in roles {
            let device = monado.ipc.device_from_role(mnd_role);
            if let Ok(mut device) = device
                && !seen.contains(&device.index)
            {
                seen.push(device.index);
                Self::update_device_battery_status(&mut device, wlx_role, &mut app.input_state);
            }
        }
        if let Ok(devices) = monado.ipc.devices() {
            for mut device in devices {
                if !seen.contains(&device.index) {
                    let role = if device.name_id >= 4 && device.name_id <= 8 {
                        TrackedDeviceRole::Tracker
                    } else {
                        TrackedDeviceRole::None
                    };
                    Self::update_device_battery_status(&mut device, role, &mut app.input_state);
                }
            }
        }

        app.input_state.devices.sort_by(|a, b| {
            u8::from(a.soc.is_none())
                .cmp(&u8::from(b.soc.is_none()))
                .then((a.role as u8).cmp(&(b.role as u8)))
                .then(a.soc.unwrap_or(999.).total_cmp(&b.soc.unwrap_or(999.)))
        });

        old_len != app.input_state.devices.len()
    }
}

impl OpenXrHandTracking {
    fn new(xr: &XrState) -> Option<Self> {
        let left = match xr.session.create_hand_tracker(xr::Hand::LEFT) {
            Ok(tracker) => {
                log::info!("OpenXR left hand tracker created successfully.");
                Some(tracker)
            }
            Err(err) => {
                log::warn!("OpenXR left hand tracker unavailable: {err:?}");
                None
            }
        };
        let right = match xr.session.create_hand_tracker(xr::Hand::RIGHT) {
            Ok(tracker) => {
                log::info!("OpenXR right hand tracker created successfully.");
                Some(tracker)
            }
            Err(err) => {
                log::warn!("OpenXR right hand tracker unavailable: {err:?}");
                None
            }
        };

        match (left, right) {
            (Some(left), Some(right)) => Some(Self {
                trackers: [left, right],
                logged_joint_success: [false, false],
                last_missing_joint_log: [Instant::now(), Instant::now()],
                last_joint_error_log: [Instant::now(), Instant::now()],
                interaction_enabled: true,
                palms_up_since: None,
                last_toggle: Instant::now(),
                last_toggle_progress_stage: 0,
            }),
            _ => {
                log::warn!(
                    "OpenXR hand trackers unavailable; continuing without XR_EXT_hand_tracking"
                );
                None
            }
        }
    }

    fn update(&mut self, state: &mut AppState, xr: &XrState) -> bool {
        let mut any_tracked = false;
        let mut derived_inputs: [Option<(bool, DerivedHandInput)>; 2] = from_fn(|_| None);

        for (idx, tracker) in self.trackers.iter().enumerate() {
            let pointer = &mut state.input_state.pointers[idx];
            let had_pose = pointer.tracked;
            any_tracked |= had_pose;

            let joints = match xr
                .stage
                .locate_hand_joints(tracker, xr.predicted_display_time)
            {
                Ok(Some(joints)) => joints,
                Ok(None) => {
                    if self.last_missing_joint_log[idx].elapsed() >= Duration::from_secs(2) {
                        log::info!(
                            "OpenXR {} hand tracker returned no joints this frame (pose source already tracked: {}).",
                            hand_name(idx),
                            had_pose
                        );
                        self.last_missing_joint_log[idx] = Instant::now();
                    }
                    continue;
                }
                Err(err) => {
                    if self.last_joint_error_log[idx].elapsed() >= Duration::from_secs(2) {
                        log::warn!(
                            "OpenXR {} hand joint locate failed: {err:?} (pose source already tracked: {}).",
                            hand_name(idx),
                            had_pose
                        );
                        self.last_joint_error_log[idx] = Instant::now();
                    }
                    continue;
                }
            };

            let Some(derived) = derive_hand_input_from_joints(&joints) else {
                if self.last_missing_joint_log[idx].elapsed() >= Duration::from_secs(2) {
                    log::info!(
                        "OpenXR {} hand joints were present but lacked the required valid positions for pose derivation.",
                        hand_name(idx)
                    );
                    self.last_missing_joint_log[idx] = Instant::now();
                }
                continue;
            };

            if !self.logged_joint_success[idx] {
                log::info!(
                    "OpenXR {} hand joints are valid. pose_fallback={} pinch_strength={:.2} grab_strength={:.2} palm_up={}",
                    hand_name(idx),
                    !had_pose,
                    derived.pinch_strength,
                    derived.grab_strength,
                    derived.palm_up
                );
                self.logged_joint_success[idx] = true;
            }

            derived_inputs[idx] = Some((had_pose, derived));
            any_tracked = true;
        }

        self.update_interaction_toggle(state, &derived_inputs);

        for (idx, maybe_derived) in derived_inputs.into_iter().enumerate() {
            let Some((had_pose, derived)) = maybe_derived else {
                continue;
            };
            let pointer = &mut state.input_state.pointers[idx];
            apply_derived_hand_input(pointer, &derived, !had_pose, self.interaction_enabled);
        }

        any_tracked
    }

    fn update_interaction_toggle(
        &mut self,
        state: &mut AppState,
        derived_inputs: &[Option<(bool, DerivedHandInput)>; 2],
    ) {
        let both_palms_up = derived_inputs.iter().all(|entry| {
            entry.as_ref().is_some_and(|(_, derived)| {
                derived.palm_up && derived.grab_strength < 0.15 && derived.pinch_strength < 0.15
            })
        });

        if both_palms_up {
            let since = self.palms_up_since.get_or_insert_with(Instant::now);
            let elapsed = since.elapsed();
            let progress_stage = ((elapsed.as_millis() / 350).min(4)) as u8;
            if progress_stage > 0 && progress_stage != self.last_toggle_progress_stage {
                self.last_toggle_progress_stage = progress_stage;
                let icon = match progress_stage {
                    1 => "◔",
                    2 => "◑",
                    3 => "◕",
                    _ => "●",
                };
                Toast::new(
                    ToastTopic::DesktopNotification,
                    format!("{icon} Hand toggle"),
                    "Hold both palms up".into(),
                )
                .with_timeout(0.45)
                .submit(state);
            }
            if elapsed >= Duration::from_millis(1400)
                && self.last_toggle.elapsed() >= Duration::from_secs(2)
            {
                self.interaction_enabled = !self.interaction_enabled;
                self.palms_up_since = None;
                self.last_toggle = Instant::now();
                self.last_toggle_progress_stage = 0;

                for pointer in &mut state.input_state.pointers {
                    pointer.interaction_enabled = self.interaction_enabled;
                    pointer.now.click = false;
                    pointer.now.grab = false;
                    pointer.now.scroll_x = 0.0;
                    pointer.now.scroll_y = 0.0;
                    pointer.now.alt_click = false;
                    pointer.now.move_mouse = false;
                    pointer.now.click_modifier_right = false;
                    pointer.now.click_modifier_middle = false;
                    pointer.now.show_hide = false;
                    pointer.now.toggle_dashboard = false;
                    pointer.now.space_drag = false;
                    pointer.now.space_rotate = false;
                    pointer.now.space_reset = false;
                }

                let state_text = if self.interaction_enabled {
                    "enabled"
                } else {
                    "disabled"
                };
                log::info!("OpenXR hand interaction toggled {state_text} via palms-up gesture.");
                Toast::new(
                    ToastTopic::DesktopNotification,
                    "Hand input".into(),
                    format!("Hand interaction {state_text}"),
                )
                .submit(state);
            }
        } else {
            self.palms_up_since = None;
            self.last_toggle_progress_stage = 0;
        }
    }
}

fn hand_name(idx: usize) -> &'static str {
    match idx {
        0 => "left",
        1 => "right",
        _ => "unknown",
    }
}

fn apply_derived_hand_input(
    pointer: &mut Pointer,
    derived: &DerivedHandInput,
    update_pose: bool,
    interaction_enabled: bool,
) {
    if update_pose {
        pointer.raw_pose = derived.pose;
        pointer.pose = derived.pose;
        pointer.tracked = true;
    }

    pointer.interaction_enabled = interaction_enabled;
    pointer.handsfree = false;
    pointer.now.scroll_x = 0.0;
    pointer.now.scroll_y = 0.0;
    pointer.now.alt_click = false;
    pointer.now.move_mouse = false;
    pointer.now.click_modifier_right = false;
    pointer.now.click_modifier_middle = false;
    pointer.now.show_hide = false;
    pointer.now.toggle_dashboard = false;
    pointer.now.space_drag = false;
    pointer.now.space_rotate = false;
    pointer.now.space_reset = false;
    pointer.now.click = interaction_enabled
        && derived.pinch_strength >= if pointer.before.click { 0.65 } else { 0.78 }
        && derived.grab_strength < 0.65;
    pointer.now.grab = interaction_enabled
        && derived.grab_strength >= if pointer.before.grab { 0.60 } else { 0.72 };
}

fn joint_position(joints: &xr::HandJointLocations, joint: xr::HandJoint) -> Option<Vec3A> {
    let joint = joints[joint.into_raw() as usize];
    if !joint
        .location_flags
        .contains(xr::SpaceLocationFlags::POSITION_VALID)
    {
        return None;
    }

    Some(Vec3A::new(
        joint.pose.position.x,
        joint.pose.position.y,
        joint.pose.position.z,
    ))
}

fn derive_hand_input_from_joints(joints: &xr::HandJointLocations) -> Option<DerivedHandInput> {
    let wrist = joint_position(joints, xr::HandJoint::WRIST)?;
    let palm = joint_position(joints, xr::HandJoint::PALM)?;
    let thumb_tip = joint_position(joints, xr::HandJoint::THUMB_TIP)?;
    let index_tip = joint_position(joints, xr::HandJoint::INDEX_TIP)?;
    let middle_tip = joint_position(joints, xr::HandJoint::MIDDLE_TIP)?;
    let ring_tip = joint_position(joints, xr::HandJoint::RING_TIP)?;
    let little_tip = joint_position(joints, xr::HandJoint::LITTLE_TIP)?;
    let index_proximal = joint_position(joints, xr::HandJoint::INDEX_PROXIMAL)?;
    let little_proximal = joint_position(joints, xr::HandJoint::LITTLE_PROXIMAL)?;

    let hand_scale = (palm - wrist).length().max(0.03);
    let pose = pose_from_hand_points(
        wrist.into(),
        palm.into(),
        index_tip.into(),
        middle_tip.into(),
        ring_tip.into(),
        little_tip.into(),
        little_proximal.into(),
        index_proximal.into(),
    );

    Some(DerivedHandInput {
        pose,
        pinch_strength: pinch_strength(thumb_tip.into(), index_tip.into(), hand_scale),
        grab_strength: grab_strength(
            palm.into(),
            [
                index_tip.into(),
                middle_tip.into(),
                ring_tip.into(),
                little_tip.into(),
            ],
            hand_scale,
        ),
        palm_up: palm_up_from_points(
            wrist.into(),
            palm.into(),
            [
                index_tip.into(),
                middle_tip.into(),
                ring_tip.into(),
                little_tip.into(),
            ],
            hand_scale,
        ),
    })
}

fn palm_up_from_points(wrist: Vec3, palm: Vec3, fingertips: [Vec3; 4], hand_scale: f32) -> bool {
    let fingertips_avg = fingertips
        .into_iter()
        .fold(Vec3::ZERO, |acc, tip| acc + tip)
        / 4.0;
    let palm_above_wrist = palm.y - wrist.y > hand_scale * 0.18;
    let fingertips_above_palm = fingertips_avg.y - palm.y > hand_scale * 0.02;
    let hand_open = grab_strength(palm, fingertips, hand_scale) < 0.20;

    palm_above_wrist && fingertips_above_palm && hand_open
}

fn normalized_strength(distance: f32, closed_distance: f32, open_distance: f32) -> f32 {
    if open_distance <= closed_distance {
        return 0.0;
    }

    (1.0 - (distance - closed_distance) / (open_distance - closed_distance)).clamp(0.0, 1.0)
}

fn pinch_strength(thumb_tip: Vec3, index_tip: Vec3, hand_scale: f32) -> f32 {
    normalized_strength(
        thumb_tip.distance(index_tip) / hand_scale.max(0.001),
        0.25,
        1.2,
    )
}

fn grab_strength(palm: Vec3, fingertips: [Vec3; 4], hand_scale: f32) -> f32 {
    let curled = fingertips
        .into_iter()
        .map(|tip| normalized_strength(tip.distance(palm) / hand_scale.max(0.001), 0.55, 1.45))
        .sum::<f32>();

    (curled / 4.0).clamp(0.0, 1.0)
}

fn pose_from_hand_points(
    wrist: Vec3,
    palm: Vec3,
    index_tip: Vec3,
    middle_tip: Vec3,
    ring_tip: Vec3,
    little_tip: Vec3,
    little_proximal: Vec3,
    index_proximal: Vec3,
) -> Affine3A {
    let wrist = Vec3A::from(wrist);
    let palm = Vec3A::from(palm);
    let index_tip = Vec3A::from(index_tip);
    let middle_tip = Vec3A::from(middle_tip);
    let ring_tip = Vec3A::from(ring_tip);
    let little_tip = Vec3A::from(little_tip);
    let little_proximal = Vec3A::from(little_proximal);
    let index_proximal = Vec3A::from(index_proximal);

    let mut right = (index_proximal - little_proximal).normalize_or_zero();
    if right.length_squared() < 0.0001 {
        right = Vec3A::X;
    }

    let stable_forward = ((middle_tip + ring_tip + little_tip) / 3.0 - palm).normalize_or_zero();
    let index_forward = (index_tip - palm).normalize_or_zero();
    let mut forward = stable_forward.lerp(index_forward, 0.12).normalize_or_zero();
    if forward.length_squared() < 0.0001 {
        forward = (index_tip - wrist).normalize_or_zero();
    }
    if forward.length_squared() < 0.0001 {
        forward = Vec3A::NEG_Z;
    }

    let mut up = right.cross(forward).normalize_or_zero();
    if up.length_squared() < 0.0001 {
        up = Vec3A::Y;
    }

    right = forward.cross(up).normalize_or_zero();
    if right.length_squared() < 0.0001 {
        right = Vec3A::X;
    }

    up = right.cross(forward).normalize_or_zero();
    if up.length_squared() < 0.0001 {
        up = Vec3A::Y;
    }

    let translation = wrist.lerp(palm, 0.7);

    Affine3A {
        matrix3: Mat3A::from_cols(right, up, -forward),
        translation,
    }
}

impl OpenXrPointer {
    pub(super) fn new(xr: &XrState, source: OpenXrHandSource) -> Result<Self, xr::sys::Result> {
        let space = source
            .pose
            .create_space(&xr.session, xr::Path::NULL, xr::Posef::IDENTITY)?;

        Ok(Self { source, space })
    }

    pub(super) fn update_handsfree(
        &mut self,
        pointer: &mut Pointer,
        xr: &XrState,
        session: &AppSession,
        hmd: Affine3A,
        hmd_tracked: bool,
    ) -> anyhow::Result<()> {
        match session.config.handsfree_pointer {
            HandsfreePointer::None => return Ok(()),
            HandsfreePointer::Hmd | HandsfreePointer::HmdOnly => {
                pointer.tracked = hmd_tracked;
                pointer.raw_pose = hmd;
                pointer.pose = hmd;
                let (cur_quat, cur_pos) =
                    (Quat::from_affine3(&pointer.pose), pointer.pose.translation);

                let (new_quat, new_pos) = (Quat::from_affine3(&hmd), Vec3::from(hmd.translation));
                let lerp_factor =
                    (1.0 / (xr.fps / 100.0) * session.config.pointer_lerp_factor).clamp(0.1, 1.0);
                pointer.raw_pose = Affine3A::from_rotation_translation(new_quat, new_pos);
                pointer.pose = Affine3A::from_rotation_translation(
                    cur_quat.lerp(new_quat, lerp_factor),
                    cur_pos.lerp(new_pos.into(), lerp_factor).into(),
                );
            }
            HandsfreePointer::EyeTracking | HandsfreePointer::EyeTrackingOnly => {
                // more aggressive smoothing for eye
                self.pointer_load_pose(pointer, xr, session.config.pointer_lerp_factor * 0.5)?;
            }
        }

        pointer.interaction_enabled = true;
        pointer.handsfree = pointer.tracked;
        if matches!(
            session.config.handsfree_pointer,
            HandsfreePointer::HmdOnly | HandsfreePointer::EyeTrackingOnly
        ) {
            // skip actions
            return Ok(());
        }

        self.pointer_load_actions(pointer, xr, session)?;

        Ok(())
    }

    pub(super) fn update(
        &mut self,
        pointer: &mut Pointer,
        xr: &XrState,
        session: &AppSession,
    ) -> anyhow::Result<()> {
        pointer.interaction_enabled = true;
        pointer.handsfree = false;
        self.pointer_load_pose(pointer, xr, session.config.pointer_lerp_factor)?;
        self.pointer_load_actions(pointer, xr, session)?;

        Ok(())
    }

    fn pointer_load_pose(
        &mut self,
        pointer: &mut Pointer,
        xr: &XrState,
        lerp_factor: f32,
    ) -> anyhow::Result<()> {
        let location = self.space.locate(&xr.stage, xr.predicted_display_time)?;
        if location
            .location_flags
            .contains(xr::SpaceLocationFlags::ORIENTATION_VALID)
        {
            let (cur_quat, cur_pos) = (Quat::from_affine3(&pointer.pose), pointer.pose.translation);

            let (new_quat, new_pos) = unsafe {
                (
                    transmute::<Quaternionf, Quat>(location.pose.orientation),
                    transmute::<Vector3f, Vec3>(location.pose.position),
                )
            };
            let lerp_factor = (1.0 / (xr.fps / 100.0) * lerp_factor).clamp(0.1, 1.0);
            pointer.raw_pose = Affine3A::from_rotation_translation(new_quat, new_pos);
            pointer.pose = Affine3A::from_rotation_translation(
                cur_quat.lerp(new_quat, lerp_factor),
                cur_pos.lerp(new_pos.into(), lerp_factor).into(),
            );
            pointer.tracked = true;
        } else {
            pointer.tracked = false;
        }
        Ok(())
    }

    fn pointer_load_actions(
        &mut self,
        pointer: &mut Pointer,
        xr: &XrState,
        session: &AppSession,
    ) -> anyhow::Result<()> {
        pointer.now.click = self.source.click.state(pointer.before.click, xr, session)?;

        pointer.now.grab = self.source.grab.state(pointer.before.grab, xr, session)?;

        let scroll = self
            .source
            .scroll
            .state(&xr.session, xr::Path::NULL)?
            .current_state;

        pointer.now.scroll_x = scroll.x;
        pointer.now.scroll_y = scroll.y;

        pointer.now.alt_click =
            self.source
                .alt_click
                .state(pointer.before.alt_click, xr, session)?;

        pointer.now.show_hide =
            self.source
                .show_hide
                .state(pointer.before.show_hide, xr, session)?;

        pointer.now.click_modifier_right =
            self.source
                .modifier_right
                .state(pointer.before.click_modifier_right, xr, session)?;

        pointer.now.toggle_dashboard =
            self.source
                .toggle_dashboard
                .state(pointer.before.toggle_dashboard, xr, session)?;

        pointer.now.click_modifier_middle =
            self.source
                .modifier_middle
                .state(pointer.before.click_modifier_middle, xr, session)?;

        pointer.now.move_mouse =
            self.source
                .move_mouse
                .state(pointer.before.move_mouse, xr, session)?;

        pointer.now.space_drag =
            self.source
                .space_drag
                .state(pointer.before.space_drag, xr, session)?;

        pointer.now.space_rotate =
            self.source
                .space_rotate
                .state(pointer.before.space_rotate, xr, session)?;

        pointer.now.space_reset =
            self.source
                .space_reset
                .state(pointer.before.space_reset, xr, session)?;

        Ok(())
    }
}

// supported action types: Haptic, Posef, Vector2f, f32, bool
impl OpenXrHandSource {
    pub(super) fn new(action_set: &mut xr::ActionSet, side: &str) -> anyhow::Result<Self> {
        let action_pose = action_set.create_action::<xr::Posef>(
            &format!("{side}_hand"),
            &format!("{side} hand pose"),
            &[],
        )?;

        let action_scroll = action_set.create_action::<Vector2f>(
            &format!("{side}_scroll"),
            &format!("{side} hand scroll"),
            &[],
        )?;
        let action_haptics = action_set.create_action::<xr::Haptic>(
            &format!("{side}_haptics"),
            &format!("{side} hand haptics"),
            &[],
        )?;

        Ok(Self {
            pose: action_pose,
            click: CustomClickAction::new(action_set, "click", side)?,
            grab: CustomClickAction::new(action_set, "grab", side)?,
            scroll: action_scroll,
            alt_click: CustomClickAction::new(action_set, "alt_click", side)?,
            show_hide: CustomClickAction::new(action_set, "show_hide", side)?,
            toggle_dashboard: CustomClickAction::new(action_set, "toggle_dashboard", side)?,
            space_drag: CustomClickAction::new(action_set, "space_drag", side)?,
            space_rotate: CustomClickAction::new(action_set, "space_rotate", side)?,
            space_reset: CustomClickAction::new(action_set, "space_reset", side)?,
            modifier_right: CustomClickAction::new(action_set, "click_modifier_right", side)?,
            modifier_middle: CustomClickAction::new(action_set, "click_modifier_middle", side)?,
            move_mouse: CustomClickAction::new(action_set, "move_mouse", side)?,
            haptics: action_haptics,
        })
    }
}

fn to_paths(maybe_path_str: Option<&str>, instance: &xr::Instance) -> Option<xr::Path> {
    maybe_path_str.and_then(|s| {
        instance
            .string_to_path(s)
            .inspect_err(|_| {
                log::warn!("Invalid binding path: {s}");
            })
            .ok()
    })
}

fn is_bool(path_str: &str) -> bool {
    path_str
        .split('/')
        .next_back()
        .is_some_and(|last| matches!(last, "click" | "touch") || last.starts_with("dpad_"))
}

macro_rules! add_custom {
    ($action:expr, $field:ident, $hands:expr, $bindings:expr, $instance:expr) => {
        if let Some(action) = $action.as_ref() {
            for i in 0..3 {
                let spec = match i {
                    0 => action.left.as_ref(),
                    1 => action.right.as_ref(),
                    2 => action.handsfree.as_ref(),
                    _ => unreachable!(),
                };

                if let Some(spec) = spec {
                    let iter: Box<dyn Iterator<Item = &String>> = match spec {
                        OneOrMany::One(s) => Box::new(std::iter::once(s)),
                        OneOrMany::Many(v) => Box::new(v.iter()),
                    };

                    for s in iter {
                        if let Some(p) = to_paths(Some(s.as_str()), $instance) {
                            if is_bool(s) {
                                if action.triple_click.unwrap_or(false) {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.triple.action_bool,
                                        p,
                                    ));
                                } else if action.double_click.unwrap_or(false) {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.double.action_bool,
                                        p,
                                    ));
                                } else {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.single.action_bool,
                                        p,
                                    ));
                                }
                            } else {
                                if action.triple_click.unwrap_or(false) {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.triple.action_f32,
                                        p,
                                    ));
                                } else if action.double_click.unwrap_or(false) {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.double.action_f32,
                                        p,
                                    ));
                                } else {
                                    $bindings.push(xr::Binding::new(
                                        &$hands[i].$field.single.action_f32,
                                        p,
                                    ));
                                }
                            }
                        };
                    }
                };
            }
        }
    };
}

// TODO: rename this bad func name
macro_rules! add_custom_lr {
    ($action:expr, $field:ident, $hands:expr, $bindings:expr, $instance:expr) => {
        if let Some(action) = $action {
            for i in 0..3 {
                let spec = match i {
                    0 => action.left.as_ref(),
                    1 => action.right.as_ref(),
                    2 => action.handsfree.as_ref(),
                    _ => unreachable!(),
                };

                if let Some(spec) = spec {
                    let iter: Box<dyn Iterator<Item = &String>> = match spec {
                        OneOrMany::One(s) => Box::new(std::iter::once(s)),
                        OneOrMany::Many(v) => Box::new(v.iter()),
                    };

                    for s in iter {
                        if let Some(p) = to_paths(Some(s.as_str()), $instance) {
                            $bindings.push(xr::Binding::new(&$hands[i].$field, p));
                        }
                    }
                };
            }
        };
    };
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn suggest_bindings(instance: &xr::Instance, hands: &[&OpenXrHandSource; 3]) {
    let profiles = load_action_profiles();

    for profile in profiles {
        log::warn!("Loading profile {}", &profile.profile);

        let Ok(profile_path) = instance.string_to_path(&profile.profile) else {
            log::warn!("Profile not supported: {}", profile.profile);
            continue;
        };

        let mut bindings: Vec<xr::Binding> = vec![];

        add_custom_lr!(profile.pose, pose, hands, bindings, instance);
        add_custom_lr!(profile.haptic, haptics, hands, bindings, instance);
        add_custom_lr!(profile.scroll, scroll, hands, bindings, instance);

        add_custom!(profile.click, click, hands, bindings, instance);

        add_custom!(profile.alt_click, alt_click, hands, bindings, instance);

        add_custom!(profile.grab, grab, hands, bindings, instance);

        add_custom!(profile.show_hide, show_hide, hands, bindings, instance);

        add_custom!(
            profile.toggle_dashboard,
            toggle_dashboard,
            hands,
            bindings,
            instance
        );

        add_custom!(profile.space_drag, space_drag, hands, bindings, instance);

        add_custom!(
            profile.space_rotate,
            space_rotate,
            hands,
            bindings,
            instance
        );

        add_custom!(profile.space_reset, space_reset, hands, bindings, instance);

        add_custom!(
            profile.click_modifier_right,
            modifier_right,
            hands,
            bindings,
            instance
        );

        add_custom!(
            profile.click_modifier_middle,
            modifier_middle,
            hands,
            bindings,
            instance
        );

        add_custom!(profile.move_mouse, move_mouse, hands, bindings, instance);

        if instance
            .suggest_interaction_profile_bindings(profile_path, &bindings)
            .is_err()
        {
            log::error!("Bad bindings for {}", &profile.profile[22..]);
            log::error!("Verify config: ~/.config/wayvr/openxr_actions.json5");
        } else {
            log::debug!(
                "Bindings for {} bound successfully.",
                &profile.profile[22..]
            );
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenXrActionConfAction {
    left: Option<OneOrMany<String>>,
    right: Option<OneOrMany<String>>,
    handsfree: Option<OneOrMany<String>>,
    threshold: Option<[f32; 2]>,
    double_click: Option<bool>,
    triple_click: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenXrActionConfProfile {
    profile: String,
    pose: Option<OpenXrActionConfAction>,
    click: Option<OpenXrActionConfAction>,
    grab: Option<OpenXrActionConfAction>,
    alt_click: Option<OpenXrActionConfAction>,
    show_hide: Option<OpenXrActionConfAction>,
    toggle_dashboard: Option<OpenXrActionConfAction>,
    space_drag: Option<OpenXrActionConfAction>,
    space_rotate: Option<OpenXrActionConfAction>,
    space_reset: Option<OpenXrActionConfAction>,
    click_modifier_right: Option<OpenXrActionConfAction>,
    click_modifier_middle: Option<OpenXrActionConfAction>,
    move_mouse: Option<OpenXrActionConfAction>,
    scroll: Option<OpenXrActionConfAction>,
    haptic: Option<OpenXrActionConfAction>,
}

const DEFAULT_PROFILES: &str = include_str!("openxr_actions.json5");

fn load_action_profiles() -> Vec<OpenXrActionConfProfile> {
    let mut profiles: Vec<OpenXrActionConfProfile> =
        serde_json5::from_str(DEFAULT_PROFILES).unwrap(); // want panic

    let Some(conf) = config_io::load("openxr_actions.json5") else {
        return profiles;
    };

    match serde_json5::from_str::<Vec<OpenXrActionConfProfile>>(&conf) {
        Ok(override_profiles) => {
            for new in override_profiles {
                if let Some(i) = profiles.iter().position(|old| old.profile == new.profile) {
                    profiles[i] = new;
                } else {
                    profiles.push(new);
                }
            }
        }
        Err(e) => {
            log::error!("Failed to load openxr_actions.json5: {e}");
        }
    }

    profiles
}

#[cfg(test)]
mod tests {
    use glam::{Vec3, Vec3A};

    use super::{grab_strength, pinch_strength, pose_from_hand_points};

    #[test]
    fn pinch_strength_increases_as_thumb_and_index_move_together() {
        let far = pinch_strength(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.08, 0.0, 0.0), 0.04);
        let near = pinch_strength(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.01, 0.0, 0.0), 0.04);

        assert!(near > 0.7, "near pinch should register strongly: {near}");
        assert!(far < 0.2, "far pinch should remain weak: {far}");
    }

    #[test]
    fn grab_strength_detects_closed_hand() {
        let open = grab_strength(
            Vec3::ZERO,
            [
                Vec3::new(0.07, 0.0, -0.02),
                Vec3::new(0.02, 0.0, -0.08),
                Vec3::new(-0.01, 0.0, -0.085),
                Vec3::new(-0.04, 0.0, -0.07),
            ],
            0.05,
        );
        let closed = grab_strength(
            Vec3::ZERO,
            [
                Vec3::new(0.03, 0.0, -0.015),
                Vec3::new(0.02, -0.005, -0.02),
                Vec3::new(0.0, -0.004, -0.018),
                Vec3::new(-0.02, -0.004, -0.016),
            ],
            0.05,
        );

        assert!(closed > 0.7, "closed hand should read as grab: {closed}");
        assert!(open < 0.35, "open hand should not read as grab: {open}");
    }

    #[test]
    fn pose_from_hand_points_points_ray_toward_index_tip() {
        let pose = pose_from_hand_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -0.03),
            Vec3::new(0.0, 0.0, -0.2),
            Vec3::new(0.0, 0.0, -0.18),
            Vec3::new(-0.01, 0.0, -0.16),
            Vec3::new(-0.02, 0.0, -0.14),
            Vec3::new(-0.05, 0.0, -0.05),
            Vec3::new(0.03, 0.0, -0.04),
        );

        let forward = pose.transform_vector3a(Vec3A::NEG_Z).normalize();
        assert!(
            forward.dot(Vec3A::new(0.0, 0.0, -1.0)) > 0.95,
            "forward was {forward:?}"
        );
        assert!((pose.translation - Vec3A::new(0.0, 0.0, -0.021)).length() < 0.0001);
    }
}
