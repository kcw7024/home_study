print("naver", "kakao", "samsung", sep=";")
# naver;kakao;samsung
# sep ::  구분자, 지정한 문자로 문자열 사이에 끼워넣어져 출력됨

# 문자열 슬라이싱
lang = "python"
print(lang[0], lang[2])
# p t :: 0번쨰와 2번째 인덱스 문자만 출력

license_plate = "24가 2240"
print(license_plate[-4:])
# 2240 :: 뒤에서 4번쨰 문자열까지만 출력

string = "abababab"
print(string[::3])
# aaaa :: 시작인덱스 / 끝인덱스 / 오프셋 을 지정하여 출력

number = "000-0000-0000"
number = number.replace("-", "")
print(number)