import pytesseract
from PIL import Image
import os
from typing import List, Optional
import cv2
import numpy as np

class KoreanNovelOCR:
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        한글 웹소설 이미지 텍스트 추출기 초기화
        
        Args:
            tesseract_path: Tesseract 실행 파일 경로 (기본값: None)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        OCR 정확도 향상을 위한 이미지 전처리
        
        Args:
            image: 입력 이미지
        Returns:
            전처리된 이미지
        """
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # 이진화
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 선명도 개선
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(binary, -1, kernel)
        
        return sharpened
    
    def extract_text(self, image_path: str) -> str:
        """
        이미지에서 한글 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
        Returns:
            추출된 텍스트
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        # 이미지 전처리
        processed_image = self.preprocess_image(image)
        
        # OCR 수행 (한글 인식 설정)
        text = pytesseract.image_to_string(
            processed_image,
            lang='kor+eng',  # 한글+영어 인식
            config='--psm 6'  # 균일한 텍스트 블록으로 가정
        )
        
        return text.strip()
    
    def batch_process(self, image_dir: str, output_file: str) -> None:
        """
        디렉토리 내의 모든 이미지 처리
        
        Args:
            image_dir: 이미지 파일들이 있는 디렉토리 경로
            output_file: 결과를 저장할 텍스트 파일 경로
        """
        image_extensions = {'.png', '.jpg', '.jpeg'}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in sorted(os.listdir(image_dir)):
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    image_path = os.path.join(image_dir, filename)
                    try:
                        text = self.extract_text(image_path)
                        f.write(f"=== {filename} ===\n")
                        f.write(text)
                        f.write('\n\n')
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

def main():
    # Tesseract 설치 경로 설정 (Windows 예시)
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # OCR 객체 생성
    ocr = KoreanNovelOCR(tesseract_path)
    
    # 여러 이미지 일괄 처리
    # 'novel_images' 폴더에 있는 모든 이미지를 처리하여 'extracted_novel.txt' 파일로 저장
    ocr.batch_process('novel_images', 'extracted_novel.txt')

if __name__ == '__main__':
    main()