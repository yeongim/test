import os
import json
import shutil

# 파일들이 있는 디렉토리
directory_path = "D:\\03_MANAGEMENT_DATA\\이륜차\\43"

# 가능한 이미지 확장자 리스트
image_extensions = ['.png', '.jpg', '.PNG', '.JPG', '.JPEG', '.jpeg']

# 디렉토리의 모든 파일을 순회
for filename in os.listdir(directory_path):
    # JSON 파일인 경우
    if filename.endswith(".json"):
        json_file_path = os.path.join(directory_path, filename)

        # JSON 파일 열기
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # shapes 리스트 안의 각 객체를 순회하면서 'label' 키의 값이 'helmet'인지 확인
        helmet_exists = any(shape['label'] == 'helmet' for shape in data['shapes'])

        # 각 확장자에 대해 이미지 파일이 존재하는지 확인하고, 존재하는 경우 그 파일을 이동
        for ext in image_extensions:
            image_file_path = os.path.join(directory_path, filename.replace('.json', ext))
            if os.path.exists(image_file_path):  # 이미지 파일이 존재하는 경우에만 이동
                if helmet_exists:
                    # 헬멧 폴더가 있으면 생성
                    if not os.path.exists("D:/03_MANAGEMENT_DATA/이륜차/헬멧있음/"):
                        os.makedirs("D:/03_MANAGEMENT_DATA/이륜차/헬멧있음/")
                    # 파일 이동
                    shutil.copy(json_file_path, "D:/03_MANAGEMENT_DATA/이륜차/헬멧있음/")
                    shutil.copy(image_file_path, "D:/03_MANAGEMENT_DATA/이륜차/헬멧있음/")
                else:
                    # 헬멧 없음 폴더가 없으면 생성
                    if not os.path.exists("D:/03_MANAGEMENT_DATA/이륜차/헬멧없음/"):
                        os.makedirs("D:/03_MANAGEMENT_DATA/이륜차/헬멧없음/")
                    # 파일 이동
                    shutil.copy(json_file_path, "D:/03_MANAGEMENT_DATA/이륜차/헬멧없음/")
                    shutil.copy(image_file_path, "D:/03_MANAGEMENT_DATA/이륜차/헬멧없음/")
                break  # 이미지 파일을 찾았으므로, 더 이상 다른 확장자를 확인할 필요 없음
