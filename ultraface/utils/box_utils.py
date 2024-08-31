import torch


def generate_priors(size, clamp=True) -> torch.Tensor:
    img_size_dict = {
        128: [128, 96],
        160: [160, 120],
        320: [320, 240],
        480: [480, 360],
        640: [640, 480],
        1280: [1280, 960]
    }
    image_size = img_size_dict[size]

    feature_map_w_h_list_dict = {
        128: [[16, 8, 4, 2], [12, 6, 3, 2]],
        160: [[20, 10, 5, 3], [15, 8, 4, 2]],
        320: [[40, 20, 10, 5], [30, 15, 8, 4]],
        480: [[60, 30, 15, 8], [45, 23, 12, 6]],
        640: [[80, 40, 20, 10], [60, 30, 15, 8]],
        1280: [[160, 80, 40, 20], [120, 60, 30, 15]]
    }
    feature_map_list = feature_map_w_h_list_dict[size]

    shrinkage_list = []
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_list[i])):
            item_list.append(image_size[i] / feature_map_list[i][k])
        shrinkage_list.append(item_list)

    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]

    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
