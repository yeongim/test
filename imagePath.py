import json
import os

folder_path = "D:\\작업\\result_모의고사5"  # 폴더 경로를 적절하게 변경하세요.

# 폴더 내의 모든 파일 목록 가져오기
file_list = os.listdir(folder_path)

for file_name in file_list:
    # 파일 확장자가 .json인 파일만 처리
    if file_name.endswith('.json'):
        file_path = os.path.join(folder_path, file_name)
        
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            

       
        # 3. "imagePath" 변경
        base_name = file_name.replace('.json', '')
        for ext in ['.jpg', '.png', '.JPG', '.PNG', '.JPEG', '.jpeg']:
            if os.path.isfile(os.path.join(folder_path, base_name + ext)):
                data['imagePath'] = base_name + ext
                break
            
  
        # 변경된 데이터를 JSON 파일에 쓰기
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
            
            
            
