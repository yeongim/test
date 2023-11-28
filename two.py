import json
import os
import numpy as np
import pandas as pd
import shutil
 

# 경로 설정
folder_path ="D:\\새 폴더" # 폴더 경로
new_path = "D:\\new_test" # 저장할 폴더 경로


# 라벨에 대한 대응 테이블
label_mapping = {
   'n1': '1', 'n2': '2',
   'n3': '3', 'n4': '4', 'n5': '5', 'n6': '6',
   'n7': '7', 'n8': '8', 'n9': '9', 'n0': '0',
   'Ga': '가', 'Na': '나', 'Da': '다', 'Ra': '라',
   'Ma': '마', 'Ba': '바', 'Sa': '사', 'Ah': '아',
   'Ja': '자', 'Cha': '차', 'Ka': '카', 'Ta': '타',
   'Pa': '파', 'Ha': '하', 'Geo': '거', 'Neo': '너',
   'Deo': '더', 'Leo': '러', 'Meo': '머', 'Beo': '버',
    'Seo': '서','Eo': '어','Jeo': '저','Cheo': '처',
    'Keo': '커','Teo': '터','Peo': '퍼','Heo': '허',
    'Go': '고', 'No': '노','Do': '도','Ro': '로',
    'Mo': '모','Bo': '보','So': '소','Oh': '오',
    'Jo': '조','Cho': '초','Ko': '코','To': '토',
    'Po': '포','Ho': '호','Gu': '구','Nu': '누',
    'Du': '두','Ru': '루','Mu': '무','Bu': '부',
    'Su': '수','U': '우','Ju': '주','Chu': '추',
    'Ku': '쿠','Tu': '투','Pu': '푸','Hu': '후',
    'Geu': '그','Neu': '느','Deu': '드','Leu': '르',
    'Meu': '므','Beu': '브','Seu': '스','Eu': '으',
    'Jeu': '즈','Cheu': '츠','Keu': '크','Teu': '트',
    'Peu': '프','Heu': '흐','Gi': '기','Ni': '니',
    'Di': '디','Li': '리','Mi': '미','Bi': '비',
    'Si': '시','I': '이','Im': '임','Ji': '지',
    'Chi': '치','Ki': '키','Ti': '티','Pi': '피',
    'Hi': '히','Yuk': '육','Hae': '해','Gong': '공',
    'Guk': '국','Hap': '합','Bae': '배','Hyphen': '-',
    'Cml': '영','vSeoul': '서울','vIncheon': '인천',
    'vBuSan': '부산','vDaeGu': '대구','vGwangJu': '광주',
    'vDaeJeon': '대전','vSeJong': '세종','vGyeongGi': '경기',
    'vGangwon': '강원','vChungBuk': '충북','vChungNam': '충남',
    'vJeonNam': '전남','vJeonBuk': '전북','vGyeongBuk': '경북',
    'vGyeongNam': '경남', 'vJeJu': '제주','vUlSan': '울산',
    'vDiplomacy': '외교','hSeoul': '서울','hIncheon': '인천',
    'hBuSan': '부산', 'hDaeGu': '대구', 'hGwangJu': '광주',
    'hDaeJeon': '대전', 'hSeJong': '세종', 'hGyeongGi': '경기',
    'hGangwon': '강원', 'hChungBuk': '충북', 'hChungNam': '충남',
    'hJeonNam': '전남', 'hJeonBuk': '전북','hGyeongBuk': '경북',
    'hGyeongNam': '경남', 'hJeJu': '제주','hUlSan': '울산','hDiplomacy': '외교',
    'OpSeoul': '서울', 'OpIncheon': '인천', 'OpBuSan': '부산', 'OpDaeGu': '대구',
    'OpGwangJu': '광주', 'OpDaeJeon': '대전', 'OpSeJong': '세종','OpGyeongGi': '경기',
    'OpGangwon': '강원', 'OpChungBuk': '충북', 'OpChungNam': '충남', 'OpJeonNam': '전남',
    'OpJeonBuk': '전북', 'OpGyeongBuk': '경북', 'OpGyeongNam': '경남', 'OpJeJu': '제주',
    'OpUlSan': '울산', 'Seoul6': '서울','Incheon6': '인천', 'BuSan6': '부산', 'DaeGu6': '대구',
    'GwangJu6': '광주', 'DaeJeon6': '대전', 'SeJong6': '세종', 'GyeongGi6': '경기', 'Gangwon6': '강원',
    'ChungBuk6': '충북', 'ChungNam6': '충남', 'JeonNam6': '전남', 'JeonBuk6': '전북','GyeongBuk6': '경북','GyeongNam6': '경남',
    'JeJu6': '제주','UlSan6': '울산','Gang': '강','Gyeong': '경','Gye': '계','Gok': '곡','Gwa': '과','Gwan': '관',
    'Gwang': '광','Goe': '괴','Gun': '군','Gwi': '귀','Geum': '금','Gim': '김','Nam': '남','Nyeong': '녕','Non': '논',
    'Dan': '단','Dal': '달','Dam': '담','Dang': '당','Dae': '대','Deok': '덕','Dong': '동','Deung': '등',
    'Rang': '랑','Rae': '래','Ryeong': '령','Rye': '례','Ryong': '룡','Reung': '릉','Myeong': '명','Mok': '목',
    'Mun': '문','Mil': '밀','Baek': '백','Bong': '봉','Buk': '북','San': '산','Sam': '삼','Sang': '상',
    'Seon': '선','Seong': '성','Se': '세','Sok': '속','Song': '송','Sun': '순','Sin': '신','Sil': '실',
    'Ak': '악','An': '안','Am': '암','Yang': '양','Yeo': '여','Yeon': '연','Yeong': '영','Ye': '예','Ok': '옥',
    'Ong': '옹','Wan': '완','Wang': '왕','Yong': '용','Un': '운','Ul': '울','Won': '원','Wol': '월','Wi': '위',
    'Yu': '유','Eun': '은','Eum': '음','Eup': '읍','Ui': '의','Ik': '익','In': '인','Im': '임','Jak': '작','Jang': '장',
    'Jeon': '전','Jeong': '정','Je': '제','Jong': '종','Jung': '중','Jeung': '증','Jin': '진','Chang': '창','Cheok': '척',
    'Cheon': '천','Cheol': '철','Cheong': '청','Chun': '춘','Chung': '충','Chil': '칠','Tae': '태','Taek': '택',
    'Tong': '통','Pyeong': '평','Ham': '함','Hang': '항','Hol': '홀','Hong': '홍','Hwa': '화','Hoeng': '횡',
    'Heung': '흥'
}

# "라벨에 대한 대응 테이블"에서 제외할 라벨 추가
excluded_labels = {
    'car': '승용','truck': '트럭','bus': '버스','motorcycle': '오토바이','bicycle': '자전거',
    'plate': '번호판','human': '사람','helmet': '헬멧','kickboard': '킥보드'
}

# 함수: 번호판 정보 생성
def generate_plate_info(shapes):
    label_parts = []  # 번호판 정보의 각 구성 요소를 저장하는 리스트
    usage = ""  # 번호판의 맨 뒤에 붙이는 "영"을 저장하는 변수
    region = ""  # 지역명을 저장하는 변수

    for shape in shapes:  # 배열 내의 각 shape에 대한 루프를 시작
        label = shape["label"]  # 현재 shape의 라벨(label)을 가져옴

        if label.startswith(("v", "h", "Op")):
           region = label  # 지역에 대한 원래 라벨 사용
           if label in label_mapping:
                region = label_mapping[label]  # 라벨에 대한 매핑 값이 있으면 해당 값을 사용
           if label.endswith("6"):  # 지역명이 "6"으로 끝나면 "6"을 제외한 나머지 부분 사용
                region = region[:-1]
        elif label.startswith("type"):
            if label == "type":  # "type" 라벨이 "type"일 때만 처리
                new_label = "번호판"  # "type"이 "번호판"을 나타내므로 번호판 라벨로 설정
                label_parts.append(new_label)
        elif label in excluded_labels:
            continue  # excluded_labels에 나열된 라벨 중 하나인 경우, 해당 라벨을 무시하고 다음 라벨을 검사
        else:  # 그 외의 경우(일반 라벨인 경우)
            new_label = label_mapping.get(label, label)  # label_mapping을 사용하여 현재 라벨(label)을 새로운 라벨(new_label)로 변환
            if new_label == "cml" and usage == "":
                label_parts.append(new_label)  # new_label이 'cml'이고 usage가 비어있는 경우, 'cml'을 label_parts에 추가하고, usage를 "영"으로 설정
                usage = "영"
            else:
                if new_label == "영":  # new_label이 '영'인 경우, usage를 "영"으로 설정
                    usage = new_label
                elif new_label.isdigit() and label_parts:
                    # 현재 라벨이 숫자이고, 이미 문자열이 시작되었을 때 숫자를 추가합니다.
                    label_parts[-1] += new_label
                else:  # 그 외의 경우, new_label을 label_parts에 추가
                    label_parts.append(new_label)

    # "영"을 마지막에 붙이도록 수정
    label_parts.append(usage)

    if region:  # region 변수에 지역명이 할당되었는지 확인/
        # 지역명이 존재하면, 해당 지역명을 label_parts 리스트의 가장 앞에 추가
        label_parts.insert(0, region)

    formatted_plate = ''.join(label_parts)

    return formatted_plate

def process_label_info(label_info, category):
    df = pd.DataFrame(label_info)
    df.insert(0, 'NO', range(len(df)))  # Add a new column 'NO' with sequential numbers starting from 0

    print(f"{category}")
    print(df.to_string(index=False))

    for idx, info in enumerate(label_info, start=0):  # Start index from 0
        label_name = info['label']
        points = info['points']

        print(f"{idx}\t{label_name}\t{points[0]}\t{points[1]}\t{points[2]}\t{points[3]}")
        
def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        return data

def process_json_file(file_path):
    data = read_json_file(file_path)
    points_and_labels = []  # 점과 라벨 정보를 저장할 리스트 초기화

    # 모든 모양을 반복하여 점과 라벨 정보를 추출
    for shape in data["shapes"]:
        label = shape["label"]
        points = [(int(round(x)), int(round(y))) for x, y in shape["points"]]
        points_and_labels.append({"label": label, "points": points})

    return points_and_labels

def print_plate_info(plate_info):
    #print(f"번호판:{plate_info}")
    if plate_info:
        df_plate = pd.DataFrame(plate_info)
        #print("\n번호판 정보")
        #print(df_plate)
        
        return plate_info

# 함수: 문자 정보를 추출하는 함수
def print_character_info(character_info):
    #print(f"문자:{character_info}")
    if character_info:
        # 수정: "car" 라벨은 문자 정보로 처리하지 않도록 변경
        character_info = [info for info in character_info if info["label"] != "car"]
        df_character = pd.DataFrame(character_info)
        #print("\n문자 정보")
        #print(df_character)
        
    return character_info

# 지역 정보
def print_region_info(region_info):
    #print(f"Original Region Info: {region_info}")

    extracted_region_info = [
        entry for entry in region_info
        if isinstance(entry, dict) and entry["label"].startswith(("v", "op", "h"))
    ]
    #print(f"Extracted Region Info: {extracted_region_info}")

    df_region = pd.DataFrame(extracted_region_info)
    #print("\n지역 정보")
    #print(df_region)

    return extracted_region_info


# 숫자 정보
def print_number_info(number_info):
    if number_info:
        # Modify the filtering condition to include labels starting with 'n'
        number_info = [info for info in number_info if info["label"].startswith("n")]

        df_number = pd.DataFrame(number_info)
        #print("\n숫자 정보")
        #print(df_number)

    return number_info

def process_json_data(data):
    plate_info = []
    character_info = []
    region_info = []
    number_info = []

    for entry in data:
        label = entry["label"]
        points = entry["points"]

        if label.startswith("type"):
            plate_info.append({"label": label, "내용": "번호판", "points": points})
        elif label.startswith("n"):
            number_info.append({"label": label, "내용": label_mapping.get(label, ""), "points": points})
        elif label.startswith(("v", "op", "h")):
            region_info.append({"label": label, "내용": label_mapping.get(label, ""), "points": points})
        else:
            character_info.append({"label": label, "내용": label_mapping.get(label, ""), "points": points})

    return plate_info, character_info, region_info, number_info


def is_label_numeric(label):
    # 여기에서 레이블이 숫자로 표현되었는지 확인
    return label.isdigit()

# 포인트가 Polygon 내에 있는지 확인한다. 포인트가 폴리곤 안에 있으면 True 아니면 False를 리턴한다.
# 모든 입력 포인트는 numpy로 입력 받는다.
def PointInPolygon(tpoint, polygon):
    """
    한 점(point)이 다각형(polygon)내부에 위치하는지 외부에 위치하는지 판별하는 함수
    입력값
        polygon -> 다각형을 구성하는 리스트 정보
        point -> 판별하고 싶은 점
    출력값
        내부에 위치하면 res = 1
        외부에 위치하면 res = 0
    """
    #(tpoint)
    N = len(polygon)   # N각형을 의미
    #print(f'N:{N} tpoint:{tpoint} polygon:{polygon}')
    counter = 0
    p1 = polygon[0]
    for i in range(1, N+1):
        p2 = polygon[i%(N)]
        #print(f'=1= i:{i} tpoint:{tpoint} p1:{p1} p2:{p2}')
        #print(f'i:{i} tpoint[0]:{tpoint[0]} p1[0]:{p1[0]}, p2[0]:{p2[0]}')
        #print(f'i:{i} tpoint[1]:{tpoint[1]} p1[1]:{p1[1]}, p2[1]:{p2[1]}')
        if tpoint[1] > min(p1[1], p2[1]) and tpoint[1] <= max(p1[1], p2[1]) and tpoint[0] <= max(p1[0], p2[0]) and p1[1] != p2[1]:
            xinters = (tpoint[1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
            #print(f'=1= {xinters} = ({tpoint[1]}-{p1[1]})*({p2[0]}-{p1[0]})/({p2[1]}-{p1[1]}) + {p1[0]}' )
            if(p1[0]==p2[0] or tpoint[0]<=xinters):
                #print(f'=2= i:{i} tpoint:{tpoint} p1:{p1} p2:{p2}')
                counter += 1
        p1 = p2
        #print(f'=3= i:{i} tpoint:{tpoint} p1:{p1} p2:{p2}')
        
    
    #print(f'counter:{counter} p1:{p1} p2:{p2}')
    res=False
    if(counter==0):
        p1 = polygon[0]
        for i in range(1, N+1):
            p2 = polygon[i%(N)]
            """
            #sense 20230705
            #확인하려는 포인트가 직선위에 있는 경우 counter가 0이 되므로 이 경우는 다각형 안으로 처리한다
            #따라서 두점을 지나는 직선의 방정식에 포인트를 대입하여 0이 되는 경우에는 내부로 처리한다.
            if(p2[1]!=p1[1]):
                xinters = (tpoint[1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1]) + p1[0]
                print(f'=2= {xinters} = ({tpoint[1]}-{p1[1]})*({p2[0]}-{p1[0]})/({p2[1]}-{p1[1]}) + {p1[0]}' )
            else :
                xinters = p1[0]
                
            if(xinters != p1[0]):
                res = False
            else:
                res = True
                break
            """
            if tpoint[1] >= min(p1[1], p2[1]) and tpoint[1] <= max(p1[1], p2[1]) and tpoint[0] <= max(p1[0], p2[0]) and tpoint[0] >= min(p1[0], p2[0]):
                if(p1[0]==p2[0]):
                    if((tpoint[0]==p1[0]) or (tpoint[0]==p2[0])):
                        res = True
                        break
                elif(p1[1]==p2[1]):
                    if((tpoint[1]==p1[1]) or (tpoint[1]==p2[1])):
                        res = True
                        break
                else:
                    gradient  = (p1[0]-p2[0])/(p1[1]-p2[1])
                    
                    if(p1[1]!=tpoint[1]):
                        tp_gradient = ((p1[0]-tpoint[0])/(p1[1]-tpoint[1]))
                    else: 
                        tp_gradient = 0
                    
                    if(gradient == tp_gradient):
                        res = True
                        break
            p1 = p2    
    else :    
        if counter % 2 == 0:
            res = False  # point is outside
        else:
            res = True  # point is inside
    """           
    if (counter % 2 == 0):
        res = False  # point is outside
    else:
        res = True  # point is inside
    """       
    return res



# 폴리콘이 폴리곤 안에 겹치는지 화인한다. 겹치는 부분이 있으면 True 아니면 False를 리턴한다
# 모든 입력 포인트는 numpy로 입력 받는다.
def PolygonOverlab(spolygon, tpolygon) :
    print(f"s:{spolygon}")
    print(f"t:{tpolygon}")
    bresult = False
    
    # 주어진 한 점씩 확인
    for point in spolygon :
        
        # PointInPolygon 함수를 통해 현재 점이 다각형 내부에 위치하는지 확인
        bresult = PointInPolygon(point, tpolygon)
        
        # 하나의 점이라도 다각형 내부에 위치하면 더 이상 확인할 필요 없음
        if (bresult):
            break
    
    print(f'PolygonOverlab:{bresult}')    
    return bresult

# box 좌표를 polygon 형태로 만든다.
def box2polygon( box):
    box_x = box[:,0]
    box_y = box[:,1]
    
    min_x = np.min(box_x,axis=0)
    min_y = np.min(box_y,axis=0)
    max_x = np.max(box_x,axis=0)
    max_y = np.max(box_y,axis=0)
    
    polygon = np.array([[min_x,min_y],[max_x,min_y],[max_x,max_y],[min_x,max_y]])
    
    
    return polygon

def check_polygon_overlap(file_path):
    data = read_json_file(file_path)
    points_and_labels = data["shapes"]
    plate_info, character_info, region_info, number_info = process_json_data(points_and_labels)

    # 번호판 정보 출력
    print("번호판 정보:")
    df_plate = pd.DataFrame(plate_info)
    print(df_plate)

    # 문자 정보 출력
    print("문자 정보:")
    df_character = pd.DataFrame(character_info)
    print(df_character)

    # 지역 정보 출력
    print("지역 정보:")
    df_region = pd.DataFrame(region_info)
    print(df_region)

    # 숫자 정보 출력
    print("숫자 정보:")
    df_number = pd.DataFrame(number_info)
    print(df_number)

    # 번호판과 문자 간 겹침 여부 확인
    for plate in plate_info:
        plate_polygon = np.array(plate["points"])
        for character in character_info:
                character_polygon = np.array(character["points"])
                overlap = PolygonOverlab(character_polygon, plate_polygon)
        print(f"번호판과 문자 간 겹침 여부: {overlap}")

    # 번호판과 숫자 간 겹침 여부 확인
    for plate in plate_info:
        plate_polygon = np.array(plate["points"])
        for number in number_info:
            number_polygon = np.array(number["points"])
            overlap = PolygonOverlab(number_polygon, plate_polygon)
            
        print(f"번호판과 숫자 간 겹침 여부: {overlap}")

    # 번호판과 지역 간 겹침 여부 확인
    for plate in plate_info:
        plate_polygon = np.array(plate["points"])
        for region in region_info:
            region_polygon = np.array(region["points"])
            overlap = PolygonOverlab(region_polygon, plate_polygon)
        print(f"번호판과 지역 간 겹침 여부: {overlap}")

# 폴더 내의 모든 JSON 파일에 대해 겹침 여부 확인
def check_overlap_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    json_files = [file_name for file_name in file_list if file_name.endswith(".json")]

    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"파일: {file_path}")
        check_polygon_overlap(file_path)
        #print()


# 폴더 내의 모든 JSON 파일에 대해 겹침 여부 확인
check_overlap_in_folder("D:\\새 폴더")    

# 파일 리스트
file_list = os.listdir(folder_path)

# Plate와 숫자 좌표를 저장할 리스트
plates_and_numbers = []  

# 각 type의 카운트를 저장할 딕셔너리
type_counts = {}  

# 현재 번호판과 관련된 정보 초기화
current_numbers = []
current_characters = []
current_regions = []

# 그룹핑된 결과를 저장할 딕셔너리
grouped_results = {}

# 이미 처리한 문자 정보를 저장할 리스트
processed_char_entries = []

# 각 파일에 대해 처리
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    print("")
    print(f"file_path:{file_path}")
    # 파일이 .json 파일인 경우
    if file_name.endswith(".json"):
        # JSON 파일을 처리하고 정보를 얻어옴
        points_and_labels = process_json_file(file_path)
        print(f"points_and_labels:{points_and_labels}")
        # 얻어온 정보가 있는 경우
        if points_and_labels:
            # JSON 데이터로부터 번호판, 문자, 지역, 숫자 정보 추출
            plate_info, char_info, region_info, number_info = process_json_data(points_and_labels)

            # 각 번호판에 대해 처리
            for plate in plate_info:
                # 튜플로 변환하여 Plate 좌표 저장
                plate_coordinates = tuple(plate["points"])

                # 그룹핑된 결과를 딕셔너리에 저장
                if plate_coordinates not in grouped_results:
                    grouped_results[plate_coordinates] = {
                        "plates": [plate],  # 번호판 정보 추가
                        "numbers": [],
                        "characters": [],
                        "regions": []
                    }
                else:
                    grouped_results[plate_coordinates]["plates"].append(plate)

                # 현재 번호판과 숫자의 겹침 여부 확인
                for number_entry in number_info:
                    number_coordinates = number_entry["points"]
                    overlap = PolygonOverlab(number_coordinates, plate_coordinates)
                    if overlap:
                        grouped_results[plate_coordinates]["numbers"].append(number_entry)

                # 현재 번호판과 문자의 겹침 여부 확인
                for char_entry in char_info:
                    char_coordinates = char_entry["points"]
                    overlap = PolygonOverlab(char_coordinates, plate_coordinates)
                    if overlap:
                        grouped_results[plate_coordinates]["characters"].append(char_entry)

                # 현재 번호판과 지역의 겹침 여부 확인
                for region_entry in region_info:
                    region_coordinates = region_entry["points"]
                    overlap = PolygonOverlab(region_coordinates, plate_coordinates)
                    if overlap:
                        grouped_results[plate_coordinates]["regions"].append(region_entry)
                        
                print("길이:", len(grouped_results))
                print("그룹:", grouped_results)

# 그룹핑된 결과 출력
for plate_coordinates, plate_info in grouped_results.items():
    print(f"Plate 좌표: {plate_coordinates}")
    print("NO\tLABEL\t내용\t\tP1\t\tP2\t\tP3\t\tP4")

    # Plate 정보 출력
    for i, plate in enumerate(plate_info["plates"]):
        print(f"{i + 1}\t{plate.get('label', '')}\t{plate.get('내용', '')}\t\t{plate['points'][0]}\t{plate['points'][1]}\t{plate['points'][2]}\t{plate['points'][3]}")

    # 문자 정보 출력
    for i, char_entry in enumerate(plate_info["characters"]):
        # char_entry가 None이 아니고, "points" 리스트가 비어있지 않은 경우에만 출력
        if char_entry and "points" in char_entry and len(char_entry["points"]) >= 4:
            print(f"{i + 1 + len(plate_info['plates'])}\t{char_entry.get('label', '')}\t{char_entry.get('내용', '')}\t\t{char_entry['points'][0]}\t{char_entry['points'][1]}\t{char_entry['points'][2]}\t{char_entry['points'][3]}")

    # 지역 정보 출력
    for i, region_entry in enumerate(plate_info["regions"]):
        print(f"{i + 1 + len(plate_info['plates']) + len(plate_info['characters'])}\t{region_entry.get('label', '')}\t{region_entry.get('내용', '')}\t\t{region_entry['points'][0]}\t{region_entry['points'][1]}\t{region_entry['points'][2]}\t{region_entry['points'][3]}")

    # 숫자 정보 출력
    for i, number_entry in enumerate(plate_info["numbers"]):
        print(f"{i + 1 + len(plate_info['plates']) + len(plate_info['characters']) + len(plate_info['regions'])}\t{number_entry.get('label', '')}\t{number_entry.get('내용', '')}\t\t{number_entry['points'][0]}\t{number_entry['points'][1]}\t{number_entry['points'][2]}\t{number_entry['points'][3]}")

    print()

             
'''
                # JSON 데이터 내의 각 항목에 대해 처리
                for entry in points_and_labels:
                    label = entry["label"]
                    print(f"label:{label}")
                    points = entry["points"]
            
                    # "type"으로 시작하는 항목
                    if label.startswith("type"):
                        type_counts[label] = type_counts.get(label, 0)
                        print("갯수:", type_counts[label])
                    # "plate"인 경우, 번호판과 관련된 숫자와 겹침 여부 확인
                    elif label == "plate":
                        for number_info in number_info:
                            number_coordinates = number_info["points"]
                            overlap = PolygonOverlab(number_coordinates, plate_coordinates)
                            if overlap:
                                plates_and_numbers.append({"plate": plate_coordinates, "number": number_info})
                    # 숫자인 경우, 현재 번호판의 숫자 정보 저장
                    elif label.isdigit():
                        current_numbers.append({"number": label, "points": points})
                    # "char"로 시작하는 경우, 현재 번호판의 문자 정보 저장
                    elif label.startswith("char"):
                        current_characters.append({"character": label, "points": points})
                    # "region"으로 시작하는 경우, 현재 번호판의 지역 정보 저장
                    elif label.startswith(("v","h","op")):
                        current_regions.append({"region": label, "points": points})
'''   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
def print_matching_results_from_json(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)

    plates_and_numbers = data["results"]

    for i, result in enumerate(plates_and_numbers, start=1):
        plate_info = result.get("plate", [])
        number_info = result.get("numbers", [])

        print(f"TYPE{i}\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
        print("NO\tLABEL\t내용\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
        for j, plate in enumerate(plate_info, start=1):
            label = plate["label"]
            content = plate["내용"]
            points = plate["points"]
            print(f"{j}\t{label}\t{content}\t\t{points[0]}\t{points[1]}\t{points[2]}\t{points[3]}")

        for j, number in enumerate(number_info, start=1):
            label = number["label"]
            content = number["내용"]
            points = number["points"]
            print(f"{j}\t{label}\t{content}\t\t{points[0]}\t{points[1]}\t{points[2]}\t{points[3]}")

        print()
     '''   
'''
def print_matching_results_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    plates_and_numbers = data["results"]
    
    for i, result in enumerate(plates_and_numbers, start=1):
        plate_info = result.get("plate", [])
        number_info = result.get("numbers", [])

        print(f"TYPE{i}")
        print("NO\tLABEL\t내용\t\tP1\t\tP2\t\tP3\t\tP4")
        for j, plate in enumerate(plate_info, start=1):
            label = plate["label"]
            content = plate["내용"]
            points = plate["points"]
            print(f"{j}\t{label}\t{content}\t\t{points}")

        for j, number in enumerate(number_info, start=1):
            label = number["label"]
            content = number["내용"]
            points = number["points"]
            print(f"{j}\t{label}\t{content}\t\t{points}")
        
        print()

'''



'''

# 추가 함수: 두 다각형 간의 거리 계산
def distance_to_line(point, line):
    x, y = point
    x1, x2, y1, y2 = line
    # 점 (x, y)에서 선 (x1, x2, y1, y2)까지의 거리를 계산
    distance = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance
'''


'''
x1 = float("inf")  # 매우 큰 값으로 초기화
x2 = float("-inf")  # 매우 작은 값으로 초기화
y1 = float("inf")
y2 = float("-inf")
'''        
# 함수 혹은 블록 내부 예시 (가상의 함수명 사용)
'''
def process_polygon_data(shapes):
    s_polygon_points = []  # s폴리곤의 포인트를 저장할 리스트
    t_polygon_points = []  # t폴리곤의 포인트를 저장할 리스트
    
    for shape in shapes:
        x, y = shape["points"]
        if shape["label"].lower().startswith(("s", "t")):
            s_polygon_points.append([x, y])
        elif shape["label"].lower().startswith("n"):
            t_polygon_points.append([x, y]
        


        # 모든 모양을 반복하여 최소 및 최대 x 좌표를 찾습니다.
        for shape in points_and_labels:
            for x, y in shape["points"]:
                #print(f"x값: {x}, y값:{y}")
                if x < x1:
                    x1 = x  # 현재 x 좌표가 x1보다 작으면 업데이트
            
                if x > x2:
                    x2 = x  # 현재 x 좌표가 x2보다 크면 업데이트
                if y < y1:
                    y1 = y
                if y > y2:
                    y2 = y
                    
                    
                if is_label_numeric(shape["label"]):  # 레이블이 숫자인 경우
                   if shape["label"].startswith(("s", "t")):  # 레이블이 's' 또는 't'로 시작하는 경우
                       s_polygon_points.append([x, y])  # s폴리곤의 포인트 리스트에 추가
                   elif shape["label"].startswith("n"):  # 레이블이 'n'으로 시작하는 경우
                       t_polygon_points.append([x, y])  # t폴리곤의 포인트 리스트에 추가
                       
                       
        # s폴리곤의 x 좌표를 추출하고 정렬
        s_polygon_x_values = sorted([point[0] for point in s_polygon_points])

        # t폴리곤의 x 좌표를 추출하고 정렬
        t_polygon_x_values = sorted([point[0] for point in t_polygon_points])

        print("s_polygon_x_values:", s_polygon_x_values)
        print("t_polygon_x_values:", t_polygon_x_values)
'''              
                    
                    
                    
                    
                    
                    
                    
                    
                    
'''
            # 참조 라인 좌표 설정
            reference_line = (x1, x2, y1, y2)
            print(reference_line)

            # 모양을 참조 라인까지의 거리에 기반하여 정렬합니다.
            #data["shapes"] = sorted(data["shapes"], key=lambda shape: distance_to_line(shape['points'][0], reference_line))

            # 번호판 정보 생성
            formatted_plate = generate_plate_info(data["shapes"])


            # 번호판 정보 출력
            print(f"번호판 정보: {formatted_plate}")
'''      
