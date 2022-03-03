def convert_to_roman(num):
  # make sure is not greater than 3,999
  if(num>3999):
    return "NUMBER IS TOO HIGH"

  roman=''
  # deal with numbers greater than 1,000
  while (num>=1000):
    roman+="M"
    num-=1000
  # for 900
  if(num>=900):
    roman+='CM'
    num-=900
  # for 500
  if(num>=500):
    roman+="D"
    num-=500
  # for 400
  if(num>=400):
    roman+="CD"
    num-=400
  # for 150
  if(num>=150):
    roman+="CL"
    num-=150
  # 100
  while(num>=100):
    roman+="C"
    num-=100
  # 90
  if(num>=90):
    roman+="XC"
    num-=90
  # 50
  if(num>=50):
    roman+="L"
    num-=50
  # 10
  while(num>=10):
    roman+="X"
    num-=10
  # 9
  if(num>=9):
    roman+="IX"
    num-=9
  # 5
  if(num>=5):
    roman+="V"
    num-=5
    # 4
  if(num>=4):
    roman+="IV"
    num-=4
  if(num>=50):
    roman+="L"
    num-=50
    # 1
  if(num>=1):
    roman+="I"
    num-=1
  return roman




# get user input
num = input("Enter a numner to convert to roman numberal: ")
# send to function
roman=convert_to_roman(int(num))
print('Num: '+num+' Roman: '+roman)

# 1 = I
# 4 = IV
# 5 = V
# 6 = VI
# 10 = x
# 50 = L
# 100 = C
# 150 = CL
# 400 = CD
# 500 = D
# 900 =CM
# 1000 = M
# max=3,999