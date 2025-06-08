# 좌석표 OCR 프로젝트

본 프로젝트는 교실 좌석표 이미지에서 OCR(Optical Character Recognition)을 이용하여 학생들의 자리 배치 정보를 자동으로 추출하고, 이를 구조화된 데이터로 변환하는 시스템입니다.

## 🧾 프로젝트 개요

- **목표**: 이미지 기반 좌석표를 자동 인식하여 학생 이름 및 정보를 추출하고, 정리된 PDF 또는 엑셀 파일로 출력
- **입력**: 좌석표 이미지 파일 (예: JPG, PNG 등)
- **출력**: 학생 이름 및 위치 정보가 정리된 PDF 및 Excel 파일

## 🛠️ 사용 기술

- Python 3.8
- OpenCV
- Tesseract OCR(ddobokki/ko-trocr)
- pandas / openpyxl (엑셀 처리)

## 📌 사용 방법

1. OCR 환경 설치:
    ```bash
    pip install -r requirements.txt
    ```

2. 좌석표 이미지 준비:
    - `data/` 폴더에 이름 텍스트 파일을 저장

3. OCR 실행:
    ```bash
    python seat_ocr.py --image ./image/test_data.jpg
    ```

4. 결과 확인:
    - `./result` 폴더에 엑셀 파일로 저장된 추출 결과 확인
