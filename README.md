# 🤖 ViT 논문 분석 및 구현 프로젝트

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

**팀명: 전압기(Transformer)**

</div>

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [프로젝트 목적](#-프로젝트-목적)
- [배경 및 필요성](#-배경-및-필요성)
- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [구현 특징](#-구현-특징)
- [참고 자료](#-참고-자료)

## 🎯 프로젝트 개요

본 프로젝트는 **Vision Transformer(ViT)**의 핵심 개념을 이해하고, 이를 처음부터 직접 구현하는 것을 목표로 합니다. `nn.Transformer`를 사용하지 않고 순수하게 구현함으로써 Transformer 아키텍처의 내부 동작 원리를 깊이 있게 학습합니다.

## 🚀 프로젝트 목적

### 1. **Transformer와 Vision Transformer(ViT)의 이해**
- 최신 딥러닝 아키텍처의 원리를 깊이 있게 이해
- 이론과 실제 구현을 통한 완전한 학습

### 2. **논문 분석 능력 향상**
- 논문을 읽고 핵심 아이디어를 추출하는 능력 개발
- 향후 연구나 프로젝트에서 활용할 수 있는 기반 마련

### 3. **실습을 통한 실무 능력 강화**
- 이론뿐만 아니라 실제 코드 구현을 통한 실무 역량 함양
- 팀 협업을 통한 프로젝트 관리 경험 축적

## 🌟 배경 및 필요성

### **최신 기술 트렌드 반영**
- AI 분야는 빠르게 발전하고 있으며, 최신 기술에 대한 이해와 구현 능력이 중요
- Transformer와 ViT는 자연어 처리(NLP)와 컴퓨터 비전(CV) 분야에서 혁신적인 성과를 달성
- 최신 기술 트렌드를 학습하고 구현함으로써 기술 역량 향상

### **AI 교육의 중요성과 취업 시장의 요구**
- AI는 다양한 산업 분야에서 핵심 기술로 자리잡음
- AI 분야 취업 시장에서 Transformer와 같은 최신 아키텍처에 대한 이해와 구현 능력 요구
- 본 프로젝트를 통해 취업 시장에서의 경쟁력 향상

### **실습 중심의 학습**
- 이론뿐만 아니라 실제 코드 작성과 모델 구현을 통한 실질적 역량 개발
- 팀 프로젝트를 통한 협업 경험과 문제 해결 능력 향상

## 🛠 기술 스택

### **개발 언어**
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)

### **프레임워크**
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

### **버전 관리**
- ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) 
- ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white) 

### **커뮤니케이션**
- ![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat-square&logo=discord&logoColor=white) 
- ![Boardmix](https://img.shields.io/badge/Boardmix-ff6600?style=flat-square&logo=discord&logoColor=white)

### **문서화 및 일정 관리**
- ![Boardmix](https://img.shields.io/badge/Boardmix-ff6600?style=flat-square&logo=discord&logoColor=white)
- ![Notion](https://img.shields.io/badge/Notion-000000?style=flat-square&logo=notion&logoColor=white) 

### **환경 관리 및 배포**
- ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
- ![KubeSphere](https://img.shields.io/badge/KubeSphere-00A971?style=flat-square&logo=kubernetes&logoColor=white)

### **인프라**
- ![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat-square&logo=nvidia&logoColor=white) GPU

## 📁 프로젝트 구조

```
visionTransformer_creator/
├── 
```

### **인프라 구성**
- KubeSphere 컨테이너 플랫폼을 활용한 학습 환경 구축
- NVIDIA GPU 서버를 통한 고속 모델 학습
- Docker 컨테이너를 통한 일관된 개발 환경 유지

## ✨ 구현 특징

### **🔥 주요 특징**
- **Pure Implementation**: `nn.Transformer`를 사용하지 않고 처음부터 구현
- **교육적 목적**: 각 구성 요소의 동작 원리를 명확히 이해할 수 있도록 구현
- **모듈화**: 각 구성 요소를 독립적인 모듈로 구현하여 재사용성 향상
- **상세한 주석**: 코드의 모든 부분에 상세한 설명 추가

## 📚 참고 자료

### **주요 논문**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 원논문
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - ViT 논문

### **학습 자료**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [딥러닝을 이용한 자연어 처리 입문-트랜스포머](https://wikidocs.net/31379) - wikidocs
- [Attention/Transformer 시각화로 설명](https://www.youtube.com/watch?v=6s69XY025MU&ab_channel=%EC%9E%84%EC%BB%A4%EB%B0%8B) - youtube 임커밋
- [[딥러닝 기계 번역] Transformer: Attention Is All You Need](https://www.youtube.com/watch?v=AA621UofTUA&ab_channel=%EB%8F%99%EB%B9%88%EB%82%98) - youtube 동빈나
- [그 이름도 유명한 어텐션, 이 영상만 보면 이해 완료!](https://www.youtube.com/watch?v=_Z3rXeJahMs&ab_channel=3Blue1Brown%ED%95%9C%EA%B5%AD%EC%96%B4) - youtube 3Blue1Brown

<div align="center">

**🔥 함께 학습하고 성장하는 전압기(Transformer) 팀 🔥**

Developed by 전압기 Team

</div>