// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Presentation for wgpu renderers on drivers without `VK_EXT_acquire_drm_display`.
//!
//! Vulkan never sees the display here.
//! GBM allocates a ring of scanout buffers, and wgpu imports each as a texture via dma-buf.
//! The frame reaches the screen through the DRM page flip the OpenGL and software paths use.
//!
//! See the presentation paths in `docs/development/window-backend-integration.md`
//! for how this fits the rest of the linuxkms backend.

// cSpell: ignore ADDFB dmabuf etnaviv LCDIF SCANOUT subresource

use std::cell::Cell;

use drm::control::Device;
use i_slint_core::platform::PlatformError;
use wgpu_30 as wgpu;

use crate::drmoutput::{DrmOutput, OwnedFramebufferHandle, SharedFd};

/// Number of buffers in the ring. With [`GbmDmabufDisplay::present_after`], a frame
/// is drawn while the GPU still finishes the previous one and the one before that
/// is still on its way to the screen, so at any moment one buffer is being drawn
/// into, one waits for the GPU, one is mid-flip and one is on screen.
pub const BUFFER_COUNT: usize = 4;

/// The scanout format. `Xrgb8888` is the one format every DRM plane is required
/// to support, and its byte order matches wgpu's `Bgra8Unorm`.
const FORMAT: gbm::Format = gbm::Format::Xrgb8888;
const WGPU_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;
/// The same format again, for the modifier query, which goes through Vulkan
/// directly. Keep the three in step.
const VK_FORMAT: ash::vk::Format = ash::vk::Format::B8G8R8A8_UNORM;

struct Buffer {
    framebuffer: OwnedFramebufferHandle,
    texture: wgpu::Texture,
    /// Whether [`Self::texture`]'s image has `VK_IMAGE_TILING_LINEAR`, see
    /// [`import_linear_image`]. Otherwise a DRM format modifier decides its tiling.
    #[cfg_attr(not(skia_wgpu_30), allow(dead_code))]
    linear_tiling: bool,
    /// Dropped after the framebuffer and the texture that refer to it.
    _bo: gbm::BufferObject<()>,
}

/// A gbm buffer as a framebuffer without a modifier, for a display that takes none.
struct ImplicitLayout<'a>(&'a gbm::BufferObject<()>);

impl drm::buffer::PlanarBuffer for ImplicitLayout<'_> {
    fn size(&self) -> (u32, u32) {
        drm::buffer::PlanarBuffer::size(self.0)
    }
    fn format(&self) -> drm::buffer::DrmFourcc {
        drm::buffer::PlanarBuffer::format(self.0)
    }
    fn modifier(&self) -> Option<drm::buffer::DrmModifier> {
        None
    }
    fn pitches(&self) -> [u32; 4] {
        drm::buffer::PlanarBuffer::pitches(self.0)
    }
    fn handles(&self) -> [Option<drm::buffer::Handle>; 4] {
        drm::buffer::PlanarBuffer::handles(self.0)
    }
    fn offsets(&self) -> [u32; 4] {
        drm::buffer::PlanarBuffer::offsets(self.0)
    }
}

pub struct GbmDmabufDisplay {
    pub drm_output: DrmOutput,
    buffers: [Buffer; BUFFER_COUNT],
    /// Index into `buffers` of the buffer to render the next frame into.
    next: Cell<usize>,
    /// A frame [`Self::present_after`] or [`Self::present_tiled`] hasn't posted yet: its
    /// buffer's index, and the submission that finishes drawing it, or `None` where the copy
    /// from [`Self::tiled`] does.
    #[cfg_attr(not(skia_wgpu_30), allow(dead_code))]
    pending: Cell<Option<(usize, Option<wgpu::SubmissionIndex>)>>,
    /// The frame to render into, see [`Self::enable_tiled_frame`].
    #[cfg(skia_wgpu_30)]
    tiled: Option<super::tiled_frame::TiledFrame>,
}

impl GbmDmabufDisplay {
    pub fn new(drm_output: DrmOutput, device: &wgpu::Device) -> Result<Self, PlatformError> {
        let gbm_device = gbm::Device::new(drm_output.drm_device.clone())
            .map_err(|e| format!("Error creating gbm device: {e}"))?;

        let (width, height) = drm_output.size();

        let mut importable_modifiers = importable_modifiers(device, width, height)?;

        // A KMS driver without DRM_CAP_ADDFB2_MODIFIERS, such as i.MX8M Plus's LCDIF,
        // rejects a framebuffer that names a modifier and scans out linear buffers only.
        let display_takes_modifiers = drm::Device::get_driver_capability(
            &drm_output.drm_device,
            drm::DriverCapability::AddFB2Modifiers,
        )
        .is_ok_and(|value| value != 0);
        if !display_takes_modifiers {
            if !importable_modifiers.contains(&gbm::Modifier::Linear) {
                return Err(PlatformError::Other(format!(
                    "The display only scans out linear buffers, but Vulkan can import the scanout \
                     format only with {importable_modifiers:?}"
                )));
            }
            importable_modifiers = vec![gbm::Modifier::Linear];
        }

        let mut buffers = Vec::with_capacity(BUFFER_COUNT);
        let mut linear_fallback = None;
        for _ in 0..BUFFER_COUNT {
            let (buffer, fallback) = Self::allocate_buffer(
                &gbm_device,
                &drm_output,
                device,
                &importable_modifiers,
                display_takes_modifiers,
                width,
                height,
            )?;
            buffers.push(buffer);
            linear_fallback = linear_fallback.or(fallback);
        }
        if let Some(reason) = linear_fallback {
            eprintln!("Scanning out linear buffers, which can be slower to render into: {reason}");
        }
        let buffers: [Buffer; BUFFER_COUNT] = buffers
            .try_into()
            .unwrap_or_else(|_| unreachable!("the loop pushes exactly BUFFER_COUNT buffers"));

        Ok(Self {
            drm_output,
            buffers,
            next: Cell::new(0),
            pending: Cell::new(None),
            #[cfg(skia_wgpu_30)]
            tiled: None,
        })
    }

    fn allocate_buffer(
        gbm_device: &gbm::Device<SharedFd>,
        drm_output: &DrmOutput,
        device: &wgpu::Device,
        importable_modifiers: &[gbm::Modifier],
        display_takes_modifiers: bool,
        width: u32,
        height: u32,
    ) -> Result<(Buffer, Option<String>), PlatformError> {
        // Allocating from the modifiers Vulkan can import lets gbm pick one that
        // is also scanout-capable, which is what `SCANOUT` asks of it.
        let with_modifiers = gbm_device.create_buffer_object_with_modifiers2::<()>(
            width,
            height,
            FORMAT,
            importable_modifiers.iter().copied(),
            gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
        );

        // A gbm backend without modifier support fails that, or ignores the list and
        // answers `Invalid`: a layout of the driver's choosing, which Vulkan can't name.
        // Asking for `LINEAR` instead gives a layout Vulkan can name.
        let (bo, linear_fallback) = match with_modifiers {
            Ok(bo) if bo.modifier() != gbm::Modifier::Invalid => (bo, None),
            with_modifiers if importable_modifiers.contains(&gbm::Modifier::Linear) => {
                let reason = match with_modifiers {
                    Ok(_) => "gbm reports no DRM format modifier".to_string(),
                    Err(e) => format!("allocating with a DRM format modifier failed: {e}"),
                };
                let bo = gbm_device
                    .create_buffer_object::<()>(
                        width,
                        height,
                        FORMAT,
                        gbm::BufferObjectFlags::SCANOUT
                            | gbm::BufferObjectFlags::RENDERING
                            | gbm::BufferObjectFlags::LINEAR,
                    )
                    .map_err(|e| {
                        format!(
                            "Error allocating a linear gbm buffer object for scanout, after \
                             {reason}: {e}"
                        )
                    })?;
                (bo, Some(reason))
            }
            Ok(_) => {
                return Err(PlatformError::Other(format!(
                    "The gbm driver reports no DRM format modifier for the scanout buffer, and \
                     Vulkan can't import a linear one, only {importable_modifiers:?}"
                )));
            }
            Err(e) => {
                return Err(PlatformError::Other(format!(
                    "Error allocating a gbm buffer object for scanout among the {} modifier(s) \
                     Vulkan can import: {e}",
                    importable_modifiers.len()
                )));
            }
        };
        let reported = bo.modifier();
        let modifier =
            if reported == gbm::Modifier::Invalid { gbm::Modifier::Linear } else { reported };

        // Only claim a modifier to KMS when gbm named one and the display takes
        // modifiers. A buffer whose layout is implicit has none to pass, even where
        // `LINEAR` made it linear for the Vulkan import above.
        // Same rule as `GbmDisplay::present`.
        let handle = if !display_takes_modifiers {
            drm_output
                .drm_device
                .add_planar_framebuffer(&ImplicitLayout(&bo), drm::control::FbCmd2Flags::empty())
        } else if reported == gbm::Modifier::Invalid {
            drm_output.drm_device.add_planar_framebuffer(&bo, drm::control::FbCmd2Flags::empty())
        } else {
            drm_output.drm_device.add_planar_framebuffer(&bo, drm::control::FbCmd2Flags::MODIFIERS)
        }
        .map_err(|e| format!("Error adding gbm buffer as framebuffer: {e}"))?;
        let framebuffer = OwnedFramebufferHandle { handle, device: drm_output.drm_device.clone() };

        let (texture, linear_tiling) = import_dmabuf_texture(device, &bo, modifier)?;

        Ok((Buffer { framebuffer, texture, linear_tiling, _bo: bo }, linear_fallback))
    }

    /// The texture to render the next frame into, and its index in the ring.
    pub fn back_buffer(&self) -> (usize, &wgpu::Texture) {
        let index = self.next.get();
        (index, &self.buffers[index].texture)
    }

    /// Advances the ring, and posts the back buffer once `submission` completed: the
    /// next time this is called, or in [`Self::flush`].
    ///
    /// Posting a frame only when the next one has been drawn lets the GPU finish it
    /// while the CPU draws the next one, at the cost of a frame of latency. Call
    /// [`Self::flush`] when no frame follows right away.
    #[cfg_attr(not(skia_wgpu_30), allow(dead_code))]
    pub fn present_after(
        &self,
        device: &wgpu::Device,
        submission: wgpu::SubmissionIndex,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let previous = self.pending.replace(Some((self.next.get(), Some(submission))));
        self.next.set((self.next.get() + 1) % BUFFER_COUNT);
        match previous {
            Some(previous) => self.post(device, previous),
            None => Ok(()),
        }
    }

    /// Renders into an optimally tiled frame from now on, which [`Self::present_tiled`]
    /// copies into the back buffer, where the scanout buffers are linear. Linear images are
    /// much slower to render into on some GPUs, such as NXP's Vivante ones; Mesa's etnaviv
    /// renders into a tiled shadow as well where it has to. Returns whether it does.
    #[cfg(skia_wgpu_30)]
    pub fn enable_tiled_frame(&mut self, device: &wgpu::Device) -> Result<bool, PlatformError> {
        if !self.buffers.iter().all(|buffer| buffer.linear_tiling) {
            return Ok(false);
        }
        self.tiled = Some(super::tiled_frame::TiledFrame::new(
            device,
            self.buffers[0].texture.size(),
            WGPU_FORMAT,
            BUFFER_COUNT,
        )?);
        Ok(true)
    }

    /// The frame to render into, after [`Self::enable_tiled_frame`].
    #[cfg(skia_wgpu_30)]
    pub fn tiled_frame(&self) -> Option<&wgpu::Texture> {
        self.tiled.as_ref().map(|tiled| &tiled.texture)
    }

    /// Copies the tiled frame into the back buffer and posts it like [`Self::present_after`].
    #[cfg(skia_wgpu_30)]
    pub fn present_tiled(
        &self,
        device: &wgpu::Device,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let index = self.next.get();
        self.tiled
            .as_ref()
            .expect("a tiled frame")
            .copy_into(index, &self.buffers[index].texture)?;
        let previous = self.pending.replace(Some((index, None)));
        self.next.set((index + 1) % BUFFER_COUNT);
        match previous {
            Some(previous) => self.post(device, previous),
            None => Ok(()),
        }
    }

    /// Posts the frame [`Self::present_after`] left pending, if any.
    #[cfg_attr(not(skia_wgpu_30), allow(dead_code))]
    pub fn flush(
        &self,
        device: &wgpu::Device,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self.pending.take() {
            Some(pending) => self.post(device, pending),
            None => Ok(()),
        }
    }

    #[cfg_attr(not(skia_wgpu_30), allow(dead_code))]
    fn post(
        &self,
        device: &wgpu::Device,
        (index, submission): (usize, Option<wgpu::SubmissionIndex>),
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // KMS has no way to wait on the render: wgpu-hal exports no sync fd for
        // the plane's IN_FENCE_FD property. A wgpu submission's fence only signals
        // once everything submitted to the queue before it completed, including
        // what Skia submitted directly.
        match submission {
            Some(submission) => {
                device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })?;
            }
            #[cfg(skia_wgpu_30)]
            None => self.tiled.as_ref().expect("a copy from the tiled frame").wait(index)?,
            #[cfg(not(skia_wgpu_30))]
            None => unreachable!(),
        }
        // A flip can only be queued once the last one landed. That one also took
        // the frame before last off screen, whose buffer is drawn into next.
        self.drm_output.wait_for_page_flip();
        self.drm_output.present_framebuffer(self.buffers[index].framebuffer.handle)
    }

    /// Posts the back buffer and advances the ring. The caller must have waited
    /// for the GPU work writing it to complete.
    #[cfg_attr(not(feature = "renderer-femtovg-wgpu"), allow(dead_code))]
    pub fn present(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let index = self.next.get();
        // Waiting here rather than before the frame was drawn overlaps rendering
        // with the flip still in flight, see `BUFFER_COUNT`. A flip can only be
        // queued once the last one landed.
        self.drm_output.wait_for_page_flip();
        self.drm_output.present_framebuffer(self.buffers[index].framebuffer.handle)?;
        self.next.set((index + 1) % BUFFER_COUNT);
        Ok(())
    }
}

/// The DRM format modifiers this Vulkan device can import a dma-buf of
/// [`FORMAT`] with, as a scanout buffer of `width` x `height`.
///
/// Handing these to gbm is what makes the two sides agree: gbm knows which
/// modifiers the display can scan out, Vulkan knows which it can import, and only
/// gbm can see both once it is given the list.
fn importable_modifiers(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<Vec<gbm::Modifier>, PlatformError> {
    use ash::vk;

    // Safety: only the physical device and instance handles are read, and the
    // queries have no side effects.
    let modifiers = unsafe {
        let hal_device = crate::renderer::dmabuf::vulkan_device(device)?;
        let physical_device = hal_device.raw_physical_device();
        let properties2 = Properties2::new(hal_device.shared_instance(), physical_device)?;

        // The first call reports how many entries there are, the second fills them in.
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        properties2.format(physical_device, VK_FORMAT, &mut properties);
        let needed = needed_tiling_features(properties.format_properties.optimal_tiling_features);

        let mut entries = vec![
            vk::DrmFormatModifierPropertiesEXT::default();
            list.drm_format_modifier_count as usize
        ];
        list.p_drm_format_modifier_properties = entries.as_mut_ptr();
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        properties2.format(physical_device, VK_FORMAT, &mut properties);

        let importable = |modifier: u64| {
            let mut modifier_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                .drm_format_modifier(modifier)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(VK_FORMAT)
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(VK_USAGE)
                .push_next(&mut modifier_info)
                .push_next(&mut external_info);
            let mut external_properties = vk::ExternalImageFormatProperties::default();
            let mut properties =
                vk::ImageFormatProperties2::default().push_next(&mut external_properties);
            let supported = properties2.image_format(physical_device, &info, &mut properties);
            let extent = properties.image_format_properties.max_extent;
            supported.is_ok()
                && extent.width >= width
                && extent.height >= height
                && external_properties
                    .external_memory_properties
                    .external_memory_features
                    .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        };

        entries
            .into_iter()
            // The import passes the first plane only, so layouts with more, such as
            // compressed ones with a metadata plane, are out.
            .filter(|entry| {
                entry.drm_format_modifier_plane_count == 1
                    && entry.drm_format_modifier_tiling_features.contains(needed)
                    && importable(entry.drm_format_modifier)
            })
            .map(|entry| gbm::Modifier::from(entry.drm_format_modifier))
            .collect::<Vec<_>>()
    };

    if modifiers.is_empty() {
        return Err(PlatformError::Other(format!(
            "The Vulkan driver can import no DRM format modifier for a {width}x{height} scanout \
             buffer"
        )));
    }

    Ok(modifiers)
}

/// The format features a scanout buffer's tiling needs, given those of `optimal` tiling.
///
/// Skia blends into its render targets and copies from and into them. It also takes
/// every wrapped image for an optimally tiled one, so it may blit wherever `optimal`
/// tiling allows that, and the scanout buffer's tiling has to keep up.
fn needed_tiling_features(optimal: ash::vk::FormatFeatureFlags) -> ash::vk::FormatFeatureFlags {
    use ash::vk::FormatFeatureFlags as Features;
    Features::COLOR_ATTACHMENT
        | Features::COLOR_ATTACHMENT_BLEND
        | Features::TRANSFER_SRC
        | Features::TRANSFER_DST
        | (optimal & (Features::BLIT_SRC | Features::BLIT_DST))
}

/// The Vulkan image usages of a scanout buffer, see [`import_dmabuf_texture`].
const VK_USAGE: ash::vk::ImageUsageFlags = ash::vk::ImageUsageFlags::from_raw(
    ash::vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | ash::vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | ash::vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// The `vkGetPhysicalDevice*Properties2` queries, from `VK_KHR_get_physical_device_properties2`
/// where wgpu enabled it, as wgpu's own queries do, and otherwise from Vulkan 1.1.
enum Properties2 {
    Core(ash::Instance),
    Extension(ash::khr::get_physical_device_properties2::Instance),
}

impl Properties2 {
    /// # Safety
    /// `physical_device` must belong to `instance`.
    unsafe fn new(
        instance: &wgpu::hal::vulkan::InstanceShared,
        physical_device: ash::vk::PhysicalDevice,
    ) -> Result<Self, PlatformError> {
        use ash::khr::get_physical_device_properties2 as extension;
        // wgpu enables the extension wherever the driver offers it, whatever the version.
        if instance.extensions().contains(&extension::NAME) {
            return Ok(Self::Extension(extension::Instance::new(
                instance.entry(),
                instance.raw_instance(),
            )));
        }
        // Physical device queries follow the version of both the instance and the device.
        let device_api_version = unsafe {
            instance.raw_instance().get_physical_device_properties(physical_device).api_version
        };
        if instance.instance_api_version().min(device_api_version) >= ash::vk::API_VERSION_1_1 {
            Ok(Self::Core(instance.raw_instance().clone()))
        } else {
            Err(PlatformError::Other(
                "The Vulkan driver offers neither Vulkan 1.1 nor \
                 VK_KHR_get_physical_device_properties2, which choosing a scanout buffer's \
                 layout needs"
                    .into(),
            ))
        }
    }

    unsafe fn format(
        &self,
        physical_device: ash::vk::PhysicalDevice,
        format: ash::vk::Format,
        properties: &mut ash::vk::FormatProperties2<'_>,
    ) {
        unsafe {
            match self {
                Self::Core(instance) => instance.get_physical_device_format_properties2(
                    physical_device,
                    format,
                    properties,
                ),
                Self::Extension(instance) => instance.get_physical_device_format_properties2(
                    physical_device,
                    format,
                    properties,
                ),
            }
        }
    }

    unsafe fn image_format(
        &self,
        physical_device: ash::vk::PhysicalDevice,
        info: &ash::vk::PhysicalDeviceImageFormatInfo2<'_>,
        properties: &mut ash::vk::ImageFormatProperties2<'_>,
    ) -> ash::prelude::VkResult<()> {
        unsafe {
            match self {
                Self::Core(instance) => instance.get_physical_device_image_format_properties2(
                    physical_device,
                    info,
                    properties,
                ),
                Self::Extension(instance) => instance.get_physical_device_image_format_properties2(
                    physical_device,
                    info,
                    properties,
                ),
            }
        }
    }
}

/// Imports `bo`'s dma-buf as a wgpu texture that renders directly into the
/// memory the display scans out. Also returns whether the image has linear tiling.
fn import_dmabuf_texture(
    device: &wgpu::Device,
    bo: &gbm::BufferObject<()>,
    modifier: gbm::Modifier,
) -> Result<(wgpu::Texture, bool), PlatformError> {
    let size = wgpu::Extent3d { width: bo.width(), height: bo.height(), depth_or_array_layers: 1 };

    let hal_descriptor = wgpu::hal::TextureDescriptor {
        label: Some("Slint linuxkms scanout buffer"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WGPU_FORMAT,
        // Skia only wraps render targets that allow copying from and into them.
        // Matches `VK_USAGE`.
        usage: wgpu::TextureUses::COLOR_TARGET
            | wgpu::TextureUses::COPY_SRC
            | wgpu::TextureUses::COPY_DST,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    // Safety: the descriptor describes `bo`, and `texture_from_dmabuf_fd` takes
    // ownership of the fd exported from it.
    let (hal_texture, linear_tiling) = unsafe {
        let hal_device = crate::renderer::dmabuf::vulkan_device(device)?;
        let linear = if modifier == gbm::Modifier::Linear {
            import_linear_image(&hal_device, bo, &hal_descriptor)?
        } else {
            None
        };
        match linear {
            Some(texture) => (texture, true),
            None => {
                let fd = bo
                    .fd()
                    .map_err(|e| format!("Error exporting gbm buffer object as dma-buf: {e}"))?;
                let texture = hal_device
                    .texture_from_dmabuf_fd(
                        fd,
                        &hal_descriptor,
                        modifier.into(),
                        bo.stride() as u64,
                        bo.offset(0) as u64,
                    )
                    .map_err(|e| format!("Error importing dma-buf as a Vulkan image: {e}"))?;
                (texture, false)
            }
        }
    };

    let descriptor = wgpu::TextureDescriptor {
        label: hal_descriptor.label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WGPU_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };

    // Safety: `hal_texture` was just created from this device and matches the
    // descriptor. Its contents are undefined until the first frame is drawn.
    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &descriptor,
            wgpu::TextureUses::UNINITIALIZED,
        )
    };
    Ok((texture, linear_tiling))
}

/// Imports a linear `bo` as a `VK_IMAGE_TILING_LINEAR` image. Returns `None` where the
/// driver can't render to such an image, or lays it out differently from gbm.
///
/// `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` with `DRM_FORMAT_MOD_LINEAR` describes the
/// same memory, but NXP's Vivante driver ignores that modifier and renders tiled.
///
/// # Safety
/// `hal_descriptor` must describe `bo`.
unsafe fn import_linear_image(
    hal_device: &wgpu::hal::vulkan::Device,
    bo: &gbm::BufferObject<()>,
    hal_descriptor: &wgpu::hal::TextureDescriptor,
) -> Result<Option<wgpu::hal::vulkan::Texture>, PlatformError> {
    use ash::vk;
    use std::os::fd::{AsRawFd, IntoRawFd};

    let instance = hal_device.shared_instance().raw_instance();
    let device = hal_device.raw_device();

    let features = unsafe {
        instance.get_physical_device_format_properties(hal_device.raw_physical_device(), VK_FORMAT)
    };
    let needed = needed_tiling_features(features.optimal_tiling_features);
    if !features.linear_tiling_features.contains(needed) || bo.offset(0) != 0 {
        return Ok(None);
    }

    let mut external_memory = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(VK_FORMAT)
        .extent(vk::Extent3D { width: bo.width(), height: bo.height(), depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(VK_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_memory);
    let image = unsafe { device.create_image(&image_info, None) }
        .map_err(|e| format!("Error creating a linear Vulkan image for the scanout buffer: {e}"))?;
    let destroy_image = || unsafe { device.destroy_image(image, None) };

    let layout = unsafe {
        device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        )
    };
    if layout.offset != 0 || layout.row_pitch != u64::from(bo.stride()) {
        destroy_image();
        return Ok(None);
    }

    let fd = bo.fd().map_err(|e| {
        destroy_image();
        format!("Error exporting gbm buffer object as dma-buf: {e}")
    })?;
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let external_memory_fd = ash::khr::external_memory_fd::Device::new(instance, device);
    let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
    if let Err(e) = unsafe {
        external_memory_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd.as_raw_fd(),
            &mut fd_properties,
        )
    } {
        destroy_image();
        return Err(format!("Error querying the scanout buffer's memory types: {e}").into());
    }
    let memory_types = requirements.memory_type_bits & fd_properties.memory_type_bits;
    if memory_types == 0 {
        destroy_image();
        return Ok(None);
    }

    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(fd.as_raw_fd());
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_types.trailing_zeros())
        .push_next(&mut import_info)
        .push_next(&mut dedicated_info);
    let memory = match unsafe { device.allocate_memory(&allocate_info, None) } {
        Ok(memory) => memory,
        Err(e) => {
            destroy_image();
            return Err(format!("Error importing the scanout buffer's memory: {e}").into());
        }
    };
    // A successful import takes ownership of the fd.
    let _ = fd.into_raw_fd();

    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe { device.free_memory(memory, None) };
        destroy_image();
        return Err(format!("Error binding the scanout buffer's memory: {e}").into());
    }

    // Safety: `image` was created from this device with `hal_descriptor`'s size,
    // format and usage, and wgpu-hal destroys it and frees `memory` with the texture.
    Ok(Some(unsafe {
        hal_device.texture_from_raw(
            image,
            hal_descriptor,
            None,
            wgpu::hal::vulkan::TextureMemory::Dedicated(memory),
        )
    }))
}
