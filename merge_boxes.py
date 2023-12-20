"""
참고: [Author: 'ZFTurbo: https://kaggle.com/zfturbo'] https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf_3d.py
"""

import numpy as np
from typing import List, Tuple, Dict


def bb_intersection_over_union(A: List[int], B: List[int]) -> float:
    """
        바운딩 박스 간 iou 계산.

        Parameters:
            A: 바운딩 박스
            B: 바운딩 박스

        Return:
            iou: 바운딩 박스의 iou 계산값
    """

    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2]) 
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou


def prefilter_bbox(det_list: List[int], threshold: float) -> Dict[str, list]:
    """
        추론 레이블 별로 검출 영역의 레이블, 점수, 바운딩박스를 저장하는 함수.

        Parameters:
            det_list: 바운딩 박스 배열,
            threshold: 바운딩 박스의 최소 점수, 최소 점수보다 낮으면 제거된다

        Return:
            categorized_boxes: 레이블 별로 바운딩 박스를 맵핑한 딕셔너리
    """

    categorized_boxes = dict()
    
    for detection in det_list:
        label = detection[0]
        score = detection[1]

        if score < threshold:
            continue

        box = [int(label), float(score), int(detection[2]), int(detection[3]), int(detection[4]), int(detection[5])]

        if label not in categorized_boxes:
            categorized_boxes[label] = []

        categorized_boxes[label].append(box)

    # Sort each list in dict and transform it to numpy array
    for k in categorized_boxes:
        current_boxes = np.array(categorized_boxes[k])
        categorized_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]] # score값이 큰 순서대로 역순 정렬

    return categorized_boxes


def get_merged_box(overlapped_boxes: List[list], conf_type: str) -> np.ndarray:
    """
        같은 객체에서 검출된 바운딩 박스를 통합하는 함수.

        Parameters:
            overlapped_boxes: 교차 영역이 있는 바운딩 박스끼리 모은 배열,
            conf_type: 바운딩 박스 병합 시 score 계산 방식

        Return:
            merged_box: 다차원 배열 형식의 병합된 바운딩 박스
    """

    merged_box = np.zeros(6, dtype=np.float32)
    
    overlapped_boxes = np.asarray(overlapped_boxes)
    merged_box[0] = overlapped_boxes[0][0] # label

    # score 병합, 평균값 or 최대값
    conf = overlapped_boxes[:, 1]
    if conf_type == 'avg':
        merged_box[1] = sum(conf)/ len(overlapped_boxes)
    elif conf_type == 'max':
        merged_box[1] = max(conf)

    # 박스의 좌표별 최대 최소값
    merged_box[2] = min(overlapped_boxes[:, 2:][:, 0])
    merged_box[3] = min(overlapped_boxes[:, 2:][:, 1])
    merged_box[4] = max(overlapped_boxes[:, 2:][:, 2])
    merged_box[5] = max(overlapped_boxes[:, 2:][:, 3])

    return merged_box


def find_overlapped_box(boxes_list: List[list], current_box: List[int], match_iou: float) -> int:
    """
        merged_box와 같은 객체에서 검출된 바운딩 박스 중 가장 일치도가 높은 박스의 인덱스를 반환.

        Parameters:
            boxes_list: 병합 과정에서 저장된 병합된 바운딩 박스의 배열,
            current_box: 병합된 바운딩 박스 배열과 비교할 바운딩 박스,
            match_iou: 바운딩 박스 간 최소 교차 영역값

        Return:
            best_index: boxes_list 중 current_box가 가장 높게 교차된 바운딩 박스의 인덱스
    """

    best_iou = match_iou
    best_index = -1

    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != current_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], current_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index


def merge_boxes(det_list: List[list], threshold: float = 0.5, iou_threshold: float = 0.1, conf_type: str = 'avg') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        같은 객체에서 검출된 바운딩 박스를 병합하는 함수.
        (정규화된 바운딩 박스는 이미지 사이즈로 스케일 필요.)

        Parameters:
            det_list: 사물 검출 결과값 (형식: [[label:int, score:float, xmin:int, ymin:int, xmax:int, ymax:int]]),
            threshold: 병합에 사용할 객체의 최소 score,
            iou_threshold: 바운딩 박스 교차 영역의 최소 iou,
            conf_type: 바운딩 박스 병합 시 score 계산 방식

        Return:
            boxes: 병합한 바운딩 박스의 박스를 포함한 다차원 배열,
            scores: 병합한 바운딩 박스의 점수를 포함한 다차원 배열,
            labels: 병합한 바운딩 박스의 레이블를 포함한 다차원 배열
    """

    if len(det_list) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)) # 검출 결과가 없을 때
    
    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    categorized_boxes = prefilter_bbox(det_list, threshold)

    # 병합한 바운딩 박스를 저장할 리스트
    overall_boxes = []

    for label in categorized_boxes:
        boxes = categorized_boxes[label]

        overlapped_boxes = []
        merged_boxes = []

        # Clusterize boxes, 박스 병합하기
        for j in range(0, len(boxes)):
            index = find_overlapped_box(merged_boxes, boxes[j], iou_threshold)
            
            if index != -1: # 겹치는 박스가 있는 경우
                overlapped_boxes[index].append(boxes[j])
                merged_boxes[index] = get_merged_box(overlapped_boxes[index], conf_type) # 박스를 병합하고 새로 할당.
            else: # 겹치는 박스가 없는 경우
                overlapped_boxes.append([boxes[j].copy()])
                merged_boxes.append(boxes[j].copy())

        overall_boxes.append(np.array(merged_boxes))

    if not overall_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)) # 검출 결과들이 threshold보다 모두 작은 경우

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]

    return boxes, scores, labels