# Transform the coordinates from the NuScenes dataset to the coordinates obtained from the collected satellite map.

params = {'singapore-onenorth': [3.35, 446.6, -3.358, 7320],
          'singapore-queenstown': [3.351, 340.1, -3.360, 12940],
          'singapore-hollandvillage': [3.357, -1062, -3.352, 10180],
          'boston-seaport': [4.534, 1111, -4.542, 10000]}


def local2global(x, y, location):
    try:
        param = params[location]
    except:
        print("location is error")
        return
    xx = param[0] * x + param[1]
    yy = param[2] * y + param[3]
    return round(xx), round(yy)


if __name__ == '__main__':
    print(local2global(1140.2, 133.3, 'boston-seaport'))
    # (6280.6668, 9394.5514)
