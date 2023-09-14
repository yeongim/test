import os
import shutil
import json

# 디렉토리 경로 설정
directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\output\\matched" # 테스트용 파일이 있는 디렉토리 경로

# 결과를 저장할 디렉토리 설정
output_directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\output"  # 결과를 저장할 디렉토리 경로

# 출력 디렉토리 생성
output_same_directory = os.path.join(output_directory, "same")
output_dffer_directory = os.path.join(output_directory, "dffer")

for output_dir in [output_same_directory, output_dffer_directory]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 이미지 파일과 JSON 파일 목록을 가져오기
test_files = os.listdir(directory)

same_image_path_count = 0  # imagePath가 일치하는 경우
different_image_path_count = 0  # imagePath가 일치하지 않는 경우
different_image_list = []

# 3번 작업: 이미지 파일과 JSON 파일의 imagePath 일치 여부 확인 및 분리
for filename in test_files:
    if filename.lower().endswith('.json'):
        # JSON 파일의 이름에서 확장자 제거
        json_filename = os.path.splitext(filename)[0]
        
        # 매칭되는 이미지 파일의 경로 생성
        png_image_path = os.path.join(directory, json_filename + ".png")
        jpg_image_path = os.path.join(directory, json_filename + ".jpg")
        PNG_image_path = os.path.join(directory, json_filename + ".PNG")
        JPG_image_path = os.path.join(directory, json_filename + ".JPG")
        
        print(f"png 경로:{png_image_path}")
        print(f"jpg 경로:{jpg_image_path}")
        print(f"PNG 경로:{PNG_image_path}")
        print(f"JPG 경로:{JPG_image_path}")
        
        # JSON 파일의 경로 생성
        json_path = os.path.join(directory, filename)
        print(f"json_path:{json_path}")
        
        # JSON 파일의 내용 읽어오기
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            image_path_in_json = json_data.get('imagePath', '')
            print(image_path_in_json)
            json_file.close()
            
            # 이미지 파일명과 JSON 파일 내의 imagePath가 일치하는 경우
            if (
                os.path.basename(png_image_path) == os.path.basename(image_path_in_json)
                or os.path.basename(jpg_image_path) == os.path.basename(image_path_in_json)
                or os.path.basename(PNG_image_path) == os.path.basename(image_path_in_json)
                or os.path.basename(JPG_image_path) == os.path.basename(image_path_in_json)
            ):

                same_image_path_count += 1

                # 이미지 파일과 JSON 파일을 output_same_directory로 이동
                print(os.path.join(directory, filename))
                print(os.path.join(output_same_directory, filename))
                shutil.move(os.path.join(directory, filename), os.path.join(output_same_directory, filename))

                # 이미지 파일도 함께 이동
                for img_path in [png_image_path, jpg_image_path, PNG_image_path, JPG_image_path]:
                    if os.path.exists(img_path):
                        img_filename = os.path.basename(img_path)
                        shutil.move(img_path, os.path.join(output_same_directory, img_filename))

            # imagePath가 일치하지 않는 경우
            else:
                different_image_path_count += 1
                print(f"매칭 안됨: {os.path.join(directory, filename)}")
                shutil.move(os.path.join(directory, filename), os.path.join(output_dffer_directory, filename))
            for img_path in [png_image_path, jpg_image_path, PNG_image_path, JPG_image_path]:
                if os.path.exists(img_path):
                    img_filename = os.path.basename(img_path)
                    shutil.move(img_path, os.path.join(output_dffer_directory, img_filename))

        print(f"매칭 카운트:{same_image_path_count}")
        print(f"언매칭 카운트:{different_image_path_count}")

# 결과 출력
print("[ImagePath 분석] 정상", same_image_path_count)
print("\t비정상", different_image_path_count)
print("\t합계:", same_image_path_count + different_image_path_count)