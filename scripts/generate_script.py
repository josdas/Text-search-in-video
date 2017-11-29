size = 5000
end = 80000

for i in range(0, end, size):
    print('python3 process_imgs.py dir=train2014 to=train_results start={} end={}'
          .format(i, min(i + size, end)), end=' ')
    print('&', end=' ')
