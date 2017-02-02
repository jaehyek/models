def a():
  print(1)
  return 1


def b():
  return a()


a.test = 3
c = b()
print(c.test)
