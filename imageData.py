import os
import json

folder_path = "D:\\작업\\result_모의고사5"

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # `imageData`의 값을 `null`로 설정합니다. 따옴표를 사용하지 않습니다.
        if 'imageData' in data:
            data['imageData'] = None

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
