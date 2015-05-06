def sumDivisionBy(divisor,number):
  return divisor*(number/divisor)*((number/divisor)+1)/2

def solution():
  return sumDivisionBy(3,999)+sumDivisionBy(5,999)-sumDivisionBy(15,999)

print solution()