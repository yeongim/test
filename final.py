import os
import shutil



# 디렉토리 경로 설정
#directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\images_전체(이륜차포함)\images"
directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\esp"
json_directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\json"
output_directory = "C:\\Users\\duck1\\작업\\차량번호\\images_전체(이륜차포함)\\output"
#output_matched_directory = os.path.join(output_directory, "matched")
#output_unmatched_directory = os.path.join(output_directory, "unmatched")
output_matched_directory = os.path.join(output_directory, "matched2")
output_unmatched_directory = os.path.join(output_directory, "unmatched2")

# 출력 디렉토리 생성
for output_dir in [output_matched_directory, output_unmatched_directory]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 각 파일 종류별 갯수를 저장할 딕셔너리 초기화
file_counts = {
    'total': 0,
    'png': 0,
    'jpg': 0,
    'json': 0,
    'unclassified': 0
}

# 이미지 파일 목록
all_files = os.listdir(directory)

# 1번 작업: 이미지 파일 갯수 확인
for filename in all_files:
    if filename.lower().endswith('.png'):
        file_counts['png'] += 1
    elif filename.lower().endswith('.jpg'):
        file_counts['jpg'] += 1
    elif filename.lower().endswith('.json'):
        file_counts['json'] += 1
    else:
        file_counts['unclassified'] += 1

    file_counts['total'] += 1
    

# 2번 작업: json/image 쌍 확인 및 분리
# 변수 초기화
matched_json_image_count = 0  # 매칭되는 JSON 파일과 이미지 파일이 모두 있는 경우의 수
matched_image_json_count = 0
unmatched_json_image_count = 0  # 매칭되는 JSON 파일은 있지만 이미지 파일이 없는 경우의 수
unmatched_image_json_count = 0  # 매칭되는 이미지 파일은 있지만 JSON 파일이 없는 경우의 수
unmatched_image_count = 0  # 비정상 이미지 파일의 수
unmatched_image_list = []


# 이미지 파일 확장자 리스트
image_extensions = ['.jpg', '.JPG', '.png', '.PNG']

print(f"all:{all_files}")
# json_files에 있는 각 JSON 파일에 대해 처리
for target_file in all_files:
    print(f"target:{target_file}")
    print("--------------------")
    if target_file.lower().endswith('.json'):
        
        # JSON 파일의 이름에서 확장자 제거
        json_filename = os.path.splitext(target_file)[0]
        print(f"확장자 제거: {json_filename}")
        
        #json 파일의 이름만 추출
        #json_filename = os.path.basename(target_file)
        #print(f"json 파일 이름 추출:{json_filename}")
        
        # 매칭되는 이미지 파일의 경로 생성
        png_image_path = os.path.join(directory, json_filename + ".png")
        jpg_image_path = os.path.join(directory, json_filename + ".jpg")
        PNG_image_path = os.path.join(directory, json_filename + ".PNG")
        JPG_image_path = os.path.join(directory, json_filename + ".JPG")
        
        
        print(f"JSON 경로: {json_filename}")
        print(f"PNG 이미지 경로: {png_image_path}")
        print(f"JPG 이미지 경로: {jpg_image_path}")
        print(f"PNG 경로:{PNG_image_path}")
        print(f"JPG 경로:{PNG_image_path}")

        
        # JSON 파일의 경로 생성
        json_path = os.path.join(directory, target_file)
        '''
        if os.path.exists(file_path):
            print("파일이 존재합니다.")
        else:
            print("파일이 존재하지 않습니다.")
        '''
       # 파일 매칭 
        if os.path.exists(png_image_path) or os.path.exists(jpg_image_path) or \
            os.path.exists(PNG_image_path) or os.path.exists(JPG_image_path):
            matched_json_image_count += 1
            print(f"이미지 매칭: {os.path.join(output_matched_directory, target_file)}")
            
            
        else:
            unmatched_json_image_count += 1
            print(f"JSON 파일은 존재하지만 매칭되는 이미지 파일이 없습니다: {target_file}")
            unmatched_image_list.append(target_file)
            # 이미지파일
            try:
                print(f"원래 경로:{os.path.join(directory, target_file)}")
                print(f"새로운 경로: {os.path.join(output_unmatched_directory, target_file)}")
                shutil.move(os.path.join(directory, target_file), os.path.join(output_unmatched_directory, target_file))
                
            except Exception as e:
                print(f"파일 이동 중 오류 발생: {e}")
    
        print("파일 매칭",matched_json_image_count)
        print(unmatched_json_image_count)
                
# 이미지 파일 목록
all2_files = os.listdir(directory)
print(f"2번 파일:{all2_files}")  
    
for target_file in all2_files:       
    # 이미지만..
   if os.path.splitext(target_file)[1] in image_extensions:
     image_filename = os.path.splitext(target_file)[0]
     json_image_path = os.path.join(directory, image_filename + ".json")
     if os.path.exists(json_image_path):
         matched_image_json_count += 1 
         try:
             shutil.move(os.path.join(directory, target_file), os.path.join(output_matched_directory, target_file))
             shutil.move(json_image_path, os.path.join(output_matched_directory, image_filename + ".json"))
         except Exception as e:
             print(f"파일 이동 중 오류 발생: {e}")
                 
     else:
         unmatched_image_json_count += 1
         unmatched_image_list.append(target_file)
         
         print(f"2번:{os.path.join(output_unmatched_directory, target_file)}")
         try:
             shutil.move(os.path.join(directory, target_file), os.path.join(output_unmatched_directory, target_file))
         except Exception as e:
             print(f"파일 이동 중 오류 발생: {e}")
     
     print("파일디렉터리",matched_image_json_count)
     print(unmatched_image_json_count)

        


print("\n[원시데이터]")
print(f"전체 이미지 갯수: {file_counts['total']}")
print(f"PNG 파일 갯수: {file_counts['png']}")
print(f"JPG 파일 갯수: {file_counts['jpg']}")
print(f"JSON 파일 갯수: {file_counts['json']}")
print(f"미분류 파일 갯수: {file_counts['unclassified']}\n")

# 결과 출력
print("[JSON/IMAGE]")
print(f"\tjson\t{matched_json_image_count + unmatched_json_image_count}/ 이미지: {matched_json_image_count + unmatched_image_json_count}")
print(f"\t정상 이미지\t{matched_json_image_count}/ json:{matched_image_json_count}")
print(f"\t비정상\t{unmatched_json_image_count + unmatched_image_json_count}")
print(f"\t전체 갯수\t{len(all_files)}")
print(f"비정상 리스트:{unmatched_image_list}")