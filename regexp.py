import re

a = "cam-api-backend-6.9"
b = "X"
c = re.sub(r'(?<=\w\-)(\d+\.)+\d+', b, a)
print(c)