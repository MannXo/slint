// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Skia on wgpu, presenting through [`crate::display::gbmdmabufdisplay`].

// cSpell: ignore bufs dmabuf

use i_slint_core::api::PhysicalSize as PhysicalWindowSize;
use i_slint_core::graphics::RequestedGraphicsAPI;
use i_slint_core::item_rendering::ItemRenderer;
use i_slint_core::platform::PlatformError;
use i_slint_core::renderer::DrawOutcome;
use i_slint_renderer_skia::{SkiaWGPU30Renderer, SkiaWGPU30RendererExt};
use wgpu_30 as wgpu;

use super::dmabuf::{init_wgpu, wait_for_gpu};
use crate::display::RenderingRotation;
use crate::display::gbmdmabufdisplay::{BUFFER_COUNT, GbmDmabufDisplay};
use crate::display::scanout_barriers::ScanoutBarriers;
use crate::drmoutput::DrmOutput;

pub struct SkiaDmabufRendererAdapter {
    renderer: SkiaWGPU30Renderer,
    display: GbmDmabufDisplay,
    barriers: ScanoutBarriers,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: PhysicalWindowSize,
}

impl SkiaDmabufRendererAdapter {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        drm_output: DrmOutput,
        requested_graphics_api: Option<&RequestedGraphicsAPI>,
    ) -> Result<Box<dyn crate::fullscreenwindowadapter::FullscreenRenderer>, PlatformError> {
        let (instance, adapter, device, queue) = init_wgpu(requested_graphics_api)?;

        let (width, height) = drm_output.size();
        let size = PhysicalWindowSize::new(width, height);

        let display = GbmDmabufDisplay::new(drm_output, &device)?;
        let barriers = ScanoutBarriers::new(&device, &queue, BUFFER_COUNT, false)?;

        let renderer = SkiaWGPU30Renderer::new(instance, adapter, device.clone(), queue.clone())?;

        eprintln!("Using Skia renderer with wgpu, presenting dma-bufs on a DRM plane");

        Ok(Box::new(Self { renderer, display, barriers, device, queue, size }))
    }
}

impl crate::fullscreenwindowadapter::FullscreenRenderer for SkiaDmabufRendererAdapter {
    fn as_core_renderer(&self) -> &dyn i_slint_core::renderer::Renderer {
        &self.renderer
    }

    fn render_and_present(
        &self,
        rotation: RenderingRotation,
        draw_mouse_cursor_callback: &dyn Fn(&mut dyn ItemRenderer),
    ) -> Result<DrawOutcome, PlatformError> {
        let (index, texture) = self.display.back_buffer();
        self.barriers.acquire(index, texture)?;
        let drawn = self
            .renderer
            .render_to_texture_transformed(
                texture,
                rotation.degrees(),
                rotation.translation_after_rotation(self.size),
                Some(&|item_renderer| {
                    draw_mouse_cursor_callback(item_renderer);
                }),
                false,
            )
            .and_then(|()| self.barriers.release(index, texture));

        if let Err(err) = drawn {
            // A failed frame can leave barriers in flight that the next one re-records,
            // without a submitted release whose fence would cover them.
            wait_for_gpu(&self.device)?;
            return Err(err);
        }

        // Skia and the release submit to the queue directly, so this empty submission
        // is the first one known to wgpu that completes after the frame.
        let submission = self.queue.submit([]);
        self.display
            .present_after(&self.device, submission)
            .map_err(|e| format!("Error presenting dma-buf: {e}"))?;
        Ok(DrawOutcome::Success)
    }

    fn flush_pending_frame(&self) -> Result<(), PlatformError> {
        self.display
            .flush(&self.device)
            .map_err(|e| format!("Error presenting dma-buf: {e}").into())
    }

    fn size(&self) -> PhysicalWindowSize {
        self.size
    }
}
