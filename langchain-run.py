import os
from dotenv import load_dotenv
import requests
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# load .env
load_dotenv()

# OpenAI GPT-4 대화 모델 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# 대화 메모리 설정
memory = ConversationBufferMemory()

# 날씨 정보를 가져오는 함수
def get_weather_forecast(city, days=1):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={os.environ.get('OPENWEATHER_API_KEY')}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        return f"{city}의 날씨: {description}, 현재 기온: {temp}°C"
    else:
        return "날씨 정보를 가져오는 데 실패했습니다."

# 사용자 정보
user_info = {
    "여행 유형": "국내 여행",
    "여행 목적": "음식 여행, 액티비티 여행",
    "선호하는 여행 스타일": "저예산 여행",
    "여행기간": "2박 3일",
    "여행 동반자": "친구와 여행",
    "여행 시기": "여름 휴가",
    "여행 예산": "50만원 ~ 100만원",
    "선호하는 숙박": "호텔/리조트",
    "여행 이동수단": "대중교통 이용",
    "관심사": "동물/자연 체험"
}

# 날씨 정보 추가
city = "Daejeon"
weather_forecast = get_weather_forecast(city)

# 프롬프트 템플릿 설정 (한국어로)
prompt_template = ChatPromptTemplate.from_template(
    "당신은 대전 관광 도우미 챗봇입니다. "
    "다음은 사용자의 과거 여행 기록 및 선호도입니다:\n"
    "여행 유형: {여행 유형}\n"
    "여행 목적: {여행 목적}\n"
    "선호하는 여행 스타일: {선호하는 여행 스타일}\n"
    "여행기간: {여행기간}\n"
    "여행 동반자: {여행 동반자}\n"
    "여행 시기: {여행 시기}\n"
    "여행 예산: {여행 예산}\n"
    "선호하는 숙박: {선호하는 숙박}\n"
    "여행 이동수단: {여행 이동수단}\n"
    "관심사: {관심사}\n"
    f"대전의 날씨 예보는 다음과 같습니다: {weather_forecast}\n\n"
    "대전 여행 일정을 추천해야한다면, 일자별 날씨 정보를 함께 포함하여 출력하고, 일정을 추천하는데 참고해주세요.\n"
    "사용자에게 추천하는 장소나 지역이 있다면, 자세한 정보(주소 및 특징)을 제공해주세요.\n"
    "사용자의 질문: {사용자 입력}\n"
    "사용자 질문에 대한 적절한 답변을 해주세요."
)

# 챗봇과의 대화 예제
def chat_with_bot(user_input):
    # 사용자 입력을 user_info에 추가하여 전달
    user_info_with_input = user_info.copy()
    user_info_with_input["사용자 입력"] = user_input
    
    # 프롬프트 템플릿에 사용자 정보를 넣어 최종 입력 생성
    prompt = prompt_template.format(**user_info_with_input)

    # 모델 호출
    response = llm.invoke(prompt)
    
    # 메모리에 사용자 입력과 응답 저장
    memory.save_context({"input": user_input}, {"output": response.content})

    return response.content

# 사용자 입력에 따라 챗봇과 대화
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        bot_response = chat_with_bot(user_input)
        print(f"Bot: {bot_response}")