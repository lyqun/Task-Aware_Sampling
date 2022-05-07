"""PyTorch implementation of the Soft Projection block."""
'''
modified from https://github.com/itailang/SampleNet/blob/master/registration/src/soft_projection.py
'''

import torch
import torch.nn as nn
import numpy as np

import knn_cuda
from pointnet2.pointnet2_utils import grouping_operation as group_point


def knn_point(group_size, point_cloud, query_cloud):
    knn_obj = knn_cuda.KNN(k=group_size, transpose_mode=False)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx


def _axis_to_dim(axis):
    """Translate Tensorflow 'axis' to corresponding PyTorch 'dim'"""
    return {0: 0, 1: 2, 2: 3, 3: 1}.get(axis)


class SoftProjection(nn.Module):
    def __init__(
        self,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-4,
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, 3, num_orig_points), database point cloud.
            query_cloud: A `Tensor` of shape (batch_size, 3, num_query_points), query items to project or propogate to.
            point_features [optional]: A `Tensor` of shape (batch_size, num_features, num_orig_points), features to propagate.
            action [optional]: 'project', 'propagate' or 'project_and_propagate'.
        Outputs:
            Depending on 'action':
            propagated_features: A `Tensor` of shape (batch_size, num_features, num_query_points)
            projected_points: A `Tensor` of shape (batch_size, 3, num_query_points)
        """

        super().__init__()
        self._group_size = group_size

        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                initial_temperature,
                requires_grad=is_temperature_trainable,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(min_sigma, dtype=torch.float32)

    def forward(self, point_cloud, query_cloud, point_features=None, action="project"):
        point_cloud = point_cloud.contiguous()
        query_cloud = query_cloud.contiguous()

        if action == "project":
            return self.project(point_cloud, query_cloud)
        elif action == "propagate":
            return self.propagate(point_cloud, point_features, query_cloud)
        elif action == "project_and_propagate":
            return self.project_and_propagate(point_cloud, point_features, query_cloud)
        else:
            raise ValueError(
                "action should be one of the following: 'project', 'propagate', 'project_and_propagate'"
            )

    def _group_points(self, point_cloud, query_cloud, point_features=None):
        group_size = self._group_size

        # find nearest group_size neighbours in point_cloud
        dist, idx = knn_point(group_size, point_cloud, query_cloud)

        # self._dist = dist.unsqueeze(1).permute(0, 1, 3, 2) ** 2

        idx = idx.permute(0, 2, 1).type(
            torch.int32
        )  # index should be Batch x QueryPoints x K
        grouped_points = group_point(point_cloud, idx)  # B x 3 x QueryPoints x K
        grouped_features = (
            None if point_features is None else group_point(point_features, idx)
        )  # B x F x QueryPoints x K
        return grouped_points, grouped_features

    def _get_distances(self, grouped_points, query_cloud):
        deltas = grouped_points - query_cloud.unsqueeze(-1).expand_as(grouped_points)
        dist = torch.sum(deltas ** 2, dim=_axis_to_dim(3), keepdim=True) / self.sigma()
        return dist

    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature ** 2, self._min_sigma.to(device))

    def project_and_propagate(self, point_cloud, point_features, query_cloud):
        # group into (batch_size, num_query_points, group_size, 3),
        # (batch_size, num_query_points, group_size, num_features)
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return (projected_points, propagated_features)

    def propagate(self, point_cloud, point_features, query_cloud):
        grouped_points, grouped_features = self._group_points(
            point_cloud, query_cloud, point_features
        )
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))

        # get weighted average of grouped_points
        propagated_features = torch.sum(
            grouped_features * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)

        return propagated_features

    def project(self, point_cloud, query_cloud, hard=False):
        grouped_points, _ = self._group_points(point_cloud, query_cloud)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = torch.softmax(-dist, dim=_axis_to_dim(2))
        if hard:
            raise NotImplementedError

        # get weighted average of grouped_points
        weights = weights.repeat(1, 3, 1, 1)
        projected_points = torch.sum(
            grouped_points * weights, dim=_axis_to_dim(2)
        )  # (batch_size, num_query_points, num_features)
        return projected_points

