# OCR Mobile card TensorFlow Serving + Docker
OCR Mobile card
### Install docker
#### Official document: https://docs.docker.com/ 

### Pull image
Download tf serving docker:
TF Serving compiled with AVX2 FMA : https://drive.google.com/file/d/1LcTkSlntkp7qlCGbIa72wSfsqnjpLFXA/view?usp=sharing

TF Serving default : https://drive.google.com/file/d/13Fvcz48F4E1R5C8jqmdLuzdtIBI8JJI6/view?usp=sharing
### Run docker container 
AVX2 FMA version:
```
docker run -p 8500:8500 --name ocr-mobile-card-service -dit --restart always -t ocr-mobile-card:latest-avx2 --model_config_file=/models/models.conf
```
Default version:
```
docker run -p 8500:8500 --name ocr-mobile-card-service -dit --restart always -t ocr-mobile-card:latest --model_config_file=/models/models.conf
```

### Install dependencies
```
pip3 install -r requirements.txt
```
