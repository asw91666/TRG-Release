import os
import sys
import cv2
from pdb import set_trace
from .path import get_file_dir_name

def video2image(input_video_path, output_image_dir, ext='jpg'):
    """
        input_video_path: "example/demo_video/David Goggins on Controlling the Multi-Voice Dialogue in Your Mind.mp4"
        output_image_dir: "example/demo_video/David Goggins on Controlling the Multi-Voice Dialogue in Your Mind"
        ext: jpg, png, ...
    """
    os.makedirs(output_image_dir, exist_ok=True)

    # 비디오 파일 로드
    cap = cv2.VideoCapture(input_video_path)

    # 프레임 번호 초기화
    frame_num = 0

    # 프레임을 성공적으로 읽었는지 확인하는 변수
    success = True

    while success:
        # 비디오에서 프레임을 하나씩 읽기
        success, frame = cap.read()

        # 프레임을 이미지 파일로 저장
        if success:
            cv2.imwrite(os.path.join(output_image_dir, f'frame_{frame_num:05d}.{ext}'), frame)
            frame_num += 1

    # 비디오 파일 해제
    cap.release()

    print(f'Done: Video -> images !!! ')

def image2video(input_image_dir, output_video_path, frame_rate=30, width=1280, height=720, img_ext='jpg'):
    """
        input_image_dir: input image dir path
        output_video_path: output video path
    """

    # cv2.VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    img_files = get_file_dir_name(input_image_dir, target_name=f'.{img_ext}')[-1]

    # 각 이미지를 읽고 비디오에 추가
    for img_file in img_files:
        img = cv2.imread(img_file)
        img_resized = cv2.resize(img, (width, height))  # 비디오 해상도에 맞게 이미지 크기 조절
        out.write(img_resized)

    # 작업 완료 후 자원 해제
    out.release()

    print('Video conversion completed!')