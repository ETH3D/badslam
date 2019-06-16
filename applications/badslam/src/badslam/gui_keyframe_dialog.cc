// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "badslam/gui_keyframe_dialog.h"

#include <libvis/eigen.h>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QString>
#include <QTabWidget>

#include "badslam/render_window.h"
#include "badslam/trajectory_deformation.h"
#include "badslam/util.cuh"

namespace vis {

KeyframeDialog::KeyframeDialog(
    atomic<usize>* current_frame_index,
    int keyframe_index,
    const shared_ptr<Keyframe>& keyframe,
    const BadSlamConfig& config,
    BadSlam* slam,
    const shared_ptr<BadSlamRenderWindow>& render_window,
    QWidget* parent)
    : QDialog(parent),
      current_frame_index_(current_frame_index),
      keyframe_(keyframe),
      slam_(slam),
      config_(config),
      render_window_(render_window) {
  setWindowTitle(tr("Keyframe #%1 (for frame #%2)").arg(keyframe_index).arg(keyframe->frame_index()));
  setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QGridLayout* layout = new QGridLayout(this);
  int row = 0;
  
  QLabel* min_max_depth_label = new QLabel(tr("Min depth: %1, max depth: %2").arg(keyframe->min_depth()).arg(keyframe->max_depth()));
  layout->addWidget(min_max_depth_label, row, 0, 1, 2);
  ++ row;
  
  QHBoxLayout* pose_layout = new QHBoxLayout();
  QLabel* pose_label = new QLabel(tr("Pose (global_tr_frame: tx ty tz qx qy qz qw): "));
  const SE3f& global_tr_frame = keyframe->global_T_frame();
  QLineEdit* pose_edit = new QLineEdit(
      QString::number(global_tr_frame.translation().x(), 'g', 14) + " " +
      QString::number(global_tr_frame.translation().y(), 'g', 14) + " " +
      QString::number(global_tr_frame.translation().z(), 'g', 14) + " " +
      QString::number(global_tr_frame.unit_quaternion().x(), 'g', 14) + " " +
      QString::number(global_tr_frame.unit_quaternion().y(), 'g', 14) + " " +
      QString::number(global_tr_frame.unit_quaternion().z(), 'g', 14) + " " +
      QString::number(global_tr_frame.unit_quaternion().w(), 'g', 14));
  connect(pose_edit, &QLineEdit::textEdited, this, &KeyframeDialog::PoseEdited);
  pose_layout->addWidget(pose_label);
  pose_layout->addWidget(pose_edit);
  layout->addLayout(pose_layout, row, 0, 1, 2);
  ++ row;
  
  // Color
  ImageDisplayQtWindow* color_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
  color_display->SetDisplayAsWidget();
  Image<uchar4> color_image_raw(keyframe->color_buffer().width(),
                                keyframe->color_buffer().height());
  keyframe->color_buffer().DownloadAsync(/*stream*/ 0, &color_image_raw);
  Image<Vec3u8> color_image(color_image_raw.size());
  for (int y = 0; y < color_image.height(); ++ y) {
    for (int x = 0; x < color_image.width(); ++ x) {
      const uchar4& v = color_image_raw(x, y);
      color_image(x, y) = Vec3u8(v.x, v.y, v.z);
    }
  }
  color_display->SetImage(color_image);
  
  // Depth
  ImageDisplayQtWindow* depth_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
  depth_display->SetDisplayAsWidget();
  Image<u16> depth_image(keyframe->depth_buffer().width(),
                         keyframe->depth_buffer().height());
  keyframe->depth_buffer().DownloadAsync(/*stream*/ 0, &depth_image);
  depth_display->SetImage(depth_image);
  depth_display->SetBlackWhiteValues(
      keyframe->min_depth() / config.raw_to_float_depth + 0.5f,
      keyframe->max_depth() / config.raw_to_float_depth + 0.5f);
  depth_display->widget().UpdateQImage();
  depth_display->widget().update(depth_display->widget().rect());
  
  // Normals
  ImageDisplayQtWindow* normals_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
  normals_display->SetDisplayAsWidget();
  Image<u16> normals_image_raw(keyframe->normals_buffer().width(),
                               keyframe->normals_buffer().height());
  keyframe->normals_buffer().DownloadAsync(/*stream*/ 0, &normals_image_raw);
  Image<Vec3u8> normals_image(keyframe->normals_buffer().width(),
                              keyframe->normals_buffer().height());
  for (u32 y = 0; y < normals_image.height(); ++ y) {
    for (u32 x = 0; x < normals_image.width(); ++ x) {
      u16 value = normals_image_raw(x, y);
      float3 result;
      result.x = EightBitSignedToSmallFloat(value & 0x00ff);
      result.y = EightBitSignedToSmallFloat((value & 0xff00) >> 8);
      result.z = -sqrtf(std::max(0.f, 1 - result.x * result.x - result.y * result.y));
      
      normals_image(x, y) = Vec3u8(
          255.99f * 0.5f * (result.x + 1.0f),
          255.99f * 0.5f * (result.y + 1.0f),
          255.99f * 0.5f * (result.z + 1.0f));
    }
  }
  normals_display->SetImage(normals_image);
  normals_display->widget().UpdateQImage();
  normals_display->widget().update(normals_display->widget().rect());
  
  // Intensities
  ImageDisplayQtWindow* intensities_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
  intensities_display->SetDisplayAsWidget();
  Image<u8> intensities_image(color_image_raw.size());
  for (int y = 0; y < color_image.height(); ++ y) {
    for (int x = 0; x < color_image.width(); ++ x) {
      const uchar4& v = color_image_raw(x, y);
      intensities_image(x, y) = v.w;
    }
  }
  intensities_display->SetImage(intensities_image);
  
  // Radius
  ImageDisplayQtWindow* radius_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ this);
  radius_display->SetDisplayAsWidget();
  Image<u16> radius_image(keyframe->radius_buffer().width(),
                          keyframe->radius_buffer().height());
  keyframe->radius_buffer().DownloadAsync(/*stream*/ 0, &radius_image);
  Image<float> radius_image_float(radius_image.size());
  for (int y = 0; y < radius_image.height(); ++ y) {
    for (int x = 0; x < radius_image.width(); ++ x) {
      if (depth_image(x, y) & kInvalidDepthBit) {
        radius_image_float(x, y) = 0;
      } else {
        // Convert half to float
        radius_image_float(x, y) = static_cast<float>(reinterpret_cast<Eigen::half&>(radius_image(x, y)));
      }
    }
  }
  radius_display->SetImage(radius_image_float);
  radius_display->SetBlackWhiteValues(0, 0.00001);
  radius_display->widget().UpdateQImage();
  radius_display->widget().update(radius_display->widget().rect());
  
  // Tab widget with the images
  QTabWidget* tab_widget = new QTabWidget(this);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget->addTab(color_display, tr("Color"));
  tab_widget->addTab(intensities_display, tr("Intensities"));
  tab_widget->addTab(depth_display, tr("Depth"));
  tab_widget->addTab(normals_display, tr("Normals"));
  tab_widget->addTab(radius_display, tr("Radii"));
  layout->addWidget(tab_widget, row, 0, 1, 2);
  
  row += 2;
  
  setLayout(layout);
}

void KeyframeDialog::PoseEdited(const QString& text) {
  Vector3d translation;
  Quaterniond rotation;
  
  if (sscanf(text.toStdString().c_str(), "%lf %lf %lf %lf %lf %lf %lf",
      &translation[0],
      &translation[1],
      &translation[2],
      &rotation.x(),
      &rotation.y(),
      &rotation.z(),
      &rotation.w()) != 7) {
    return;
  }
  
  vector<SE3f> original_keyframe_T_global;
  RememberKeyframePoses(&slam_->direct_ba(), &original_keyframe_T_global);
  
  keyframe_->set_global_T_frame(SE3d(rotation, translation).cast<float>());
  slam_->direct_ba().UpdateKeyframeCoVisibility(keyframe_);
  
  vis::ExtrapolateAndInterpolateKeyframePoseChanges(
      config_.start_frame,
      *current_frame_index_,
      &slam_->direct_ba(),
      original_keyframe_T_global,
      slam_->rgbd_video());
  
  slam_->direct_ba().UpdateBAVisualization(0);
  render_window_->RenderFrame();
}

}
