import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
np.random.seed(11)
# np.random.seed(29)
# np.random.seed(143)

def first_pic(data, x_limits, y_limits, size=7, title = ""):
    """
    Рисует только полученные точки
    """
    figsize = np.array((abs(x_limits[0] - x_limits[1]), abs(y_limits[0] - y_limits[1])))
    figsize = figsize*(size/np.max(figsize))
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(title)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.axis("equal")
    ax.scatter(data[0],data[1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return fig


x_limits = [-10,10]
y_limits = [-5,5]
N = 6

def random_data(x_limits, y_limits,N):
    """
    Функция создает массив точек на плоскости в заданных ограничениях 
    с равномерным распределением
    ограничения (x_limits, y_limits) заданы в виде (x_min, x_max)
    Размер выходного массива (2,N)
    """
    return np.vstack((np.random.uniform(x_limits[0],x_limits[1], N),
            np.random.uniform(y_limits[0],y_limits[1], N)))


def distance(dot1, dot2):
    """
    функция расстояния между точками
    """
    return np.sqrt(np.sum((dot1- dot2)**2))

def max_distance(data):
    """
    Функция ищет индексы точек, между которыми максимальное расстояние
    """
    max_val = 0
    index = [0,0]
    # транспонируем данные для удобного обращения к точкам
    for in1, dot1 in enumerate(data.T):
        for in2, dot2 in enumerate(data.T):
            if distance(dot1, dot2)> max_val:
                max_val = distance(dot1, dot2)
                index = [in1, in2]
    # сортируем индексы по х
    return sorted(index, key = lambda ind: data[0,ind])

def rotate(A,B,C):
    """
    синус угла ABC. в точке B
    """
    # формула: (a_x b_y - a_y b_x) /|a|/|b|. a = BA, b = BC
    # подаются точки (np.array из 2 чисел)
    # z = (A[0] - B[0])*(C[1] - B[1]) - (A[1] - B[1])*(C[0] - B[0])
    # if z:
    #     return z /distance(A,B) /distance(B,C)
    # return z

    angle = np.angle((C -B)[0] + 1j*(C - B)[1]) -np.angle((A - B)[0]+ 1j*(A - B)[1])
    return np.sin(angle)



def graham_scan(data):
    #транспонируем для удобства
    dots_list = data.T

    n = len(dots_list)
    index_list= list(range(n))

    # индекс мин по x элемента
    # in_min = np.argmin(dots_list.T[0])
    in_min = np.argmin(dots_list,axis= 0)[0]
    
    # перемещаем индекс мин по x эл. в 0 позицию.
    index_list[0], index_list[in_min] = index_list[in_min], index_list[0]
    
    # сортировка по углу 
    dot = np.array([dots_list[index_list[0]][0]+1,dots_list[index_list[0]][1]])
    index_list[1:] = sorted(index_list[1:], key=lambda A: rotate(dot, dots_list[index_list[0]], dots_list[A] ))
    
    # формируем окончательный список
    res = [index_list[0], index_list[1]]
    num = 2
    while num != 0:
        # если новая точка ломает построение выпуклой оболочки, то снимаем поледнюю добавленную в список
        while rotate(dots_list[index_list[num]],dots_list[res[-1]], dots_list[res[-2]]) <=0:
            res.pop(-1)
        res.append(index_list[num])
        num = (num+1) % n


    return res


def _angle(con_pol, m,n):
    """
    Функция для rotating_calipers
    считает угол между прямыми, что образованы вершинами выпуклого многоуголькика
    первая прямая создается из 2 вершин с индексами m, m+1
    вторая прямая создается из 2 вершин с индексами n, n+1
    Номера берутся по модулю количества вершин многоугольника
    Угол - это угол от первой прямой ко второй по направлению часовой стрелки
    """
    
    mod = con_pol.shape[0]

    k1 = np.arctan2((con_pol[(m+1) % mod][1] - con_pol[m% mod][1]),(con_pol[(m+1) % mod][0] - con_pol[m% mod][0]))
    
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    
    print(k1,k2)

    k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    res = k1 - k2
    if k1 == k2:
        print('!!!!!!!!!!')
        print(k1 - k2, res)

    # res = res if k1 >= k2 else np.pi - res*np.sign(res)
    if k1 > k2:
        # res = np.pi - res*np.sign(res)
        res =res
    if k1 < k2:
        res = np.pi - np.sign(k1)*k1 - k2
    
    # res = res if res < np.pi else res - np.pi
    # res = res if res > 0 else -1*res

    print(f'k1 {k1}, k2 {k2}, res {res}')
    return res 

def first_angle(con_pol, m,n):
    """
    Функция для rotating_calipers
    считает угол между прямыми, что образованы вершинами выпуклого многоуголькика
    первая прямая создается из 2 вершин с индексами m, m+1
    вторая прямая создается из 2 вершин с индексами n, n+1
    Номера берутся по модулю количества вершин многоугольника
    Угол - это угол от первой прямой ко второй по направлению часовой стрелки
    """
    
    mod = con_pol.shape[0]

    k1 = np.arctan2((con_pol[(m) % mod][1] - con_pol[(m+1)% mod][1]),(con_pol[(m) % mod][0] - con_pol[(m+1)% mod][0]))
    # k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    # k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    res = k1 - k2
    # res = res if res > 0 else np.pi + res
    # res = res if res < np.pi else res - np.pi
    res = res if res >0  else res + 2*np.pi

    print(f'k1 {k1}, k2 {k2}, res {res}')
    return res 

def _rotating_calipers(con_pol):
    """
    Измененный Shamos's algorithm
    """
    n = len(con_pol)

    i = np.argmin(con_pol, axis=0)[0]
    j = np.argmax(con_pol, axis=0)[0]

    # if con_pol[i+1][0] - con_pol[i][0] > con_pol[j][0] - con_pol[j+1][0]:
    #     i,j = j,i

    i, j=0, 1
    while first_angle(con_pol, i, j)< np.pi:
        j+=1

    yield i, j

    current = i
    while j < n  or i < n:
        print(f'current {current}, i+1 & angle:  {i+1, angle(con_pol,current, i+1)}, j+1 & angle: {j+1, angle(con_pol,current, j+1)}')
        if angle(con_pol, current, i+1) <= angle(con_pol,current, j+1):
            j+=1
            current =j
        else:
            i+=1
            current = i
        yield i,j

        if angle(con_pol,current, i+1) == angle(con_pol,current,j+1):
            yield i+1, j
            yield i, j+1
            yield i+1, j+1

            if current == i:
                j+=1
            else:
                i+=1

def quick_max_distance(data):
    index_list = graham_scan(data)

    con_pol = data.T[index_list]
    mod = con_pol.shape[0]
    print(mod)
    max_d = 0
    for i in rotating_calipers(con_pol):
        print(i)
        i= (i[0]% mod, i[1]% mod)
        if max_d< distance(con_pol[i[0]], con_pol[i[1]]):
            max_d = distance(con_pol[i[0]], con_pol[i[1]])
            res = i
            print('res=',res)
    return index_list[res[0]], index_list[res[1]]


def rotating_calipers(con_pol):
    """
    Измененный Shamos's algorithm
    """
    n = len(con_pol)

    i0 = np.argmin(con_pol, axis=0)[0] 
    i = i0
    j = i+1

    while first_angle(con_pol, i, j)< np.pi:
        j+=1
    
    yield i, j

    current = i
    i+=1
    while j != n+1:
        # print(f'current {current}, i+1 & angle:  {i+1, angle(con_pol,current, i+1)}, j+1 & angle: {j+1, angle(con_pol,current, j+1)}')
        print(f'current {current}, i & angle:  {i, angle(con_pol,current, i)}, j & angle: {j, angle(con_pol,current, j)}')
        if angle(con_pol,current, i) == angle(con_pol,current,j):
            yield i, j
            yield i+1, j
            yield i, j+1
            # yield i+1, j+1

            i+=1
            j+=1

        
        if angle(con_pol, current, i) < angle(con_pol,current, j):
            current = i
            yield i,j
            i+=1
        else:
            current = j
            yield i,j
            j+=1



        


def area(con_pol, i,j, k):
    
    mod = con_pol.shape[0]
    
    i %=  mod
    j %= mod
    k %= mod


    a = distance(con_pol[i], con_pol[j])
    b = distance(con_pol[i], con_pol[k])
    c = distance(con_pol[j], con_pol[k])

    p = (a+b+c)/2

    return np.sqrt( p*(p-a)*(p-b)*(p-c) )


def BAD_rotating_calipers(con_pol):
    n =con_pol.shape[0]

    i0 = n-1
    i = 0
    j = i+1

    while area(con_pol, i, i+1, j+1) > area(con_pol, i, i+1, j):
        j+=1
        j0 = j
        print('test')
    while j!= i0 and (i,j)!= (j0,i0):
        i+=1
        yield i,j

        while area(con_pol, i, i+1, j+1)> area(con_pol,i, i+1, j) and (i,j)!= (j0,i0):
            j+=1
            if (i,j)!= (j0,i0):
                yield i,j
            else:
                break
        if (i,j)== (j0,i0):
            break

        if area(con_pol,j,i +1, j+1) == area(con_pol, i, i+1, j):
            if (i,j) != (j0,i0):
                yield i, j+1
            else:
                yield i+1, j
                

def test():
    k = 500

    for i in range(k):
        data = random_data(x_limits, y_limits,N)
        

        R1 = quick_max_distance(data)
        R2 = max_distance(data)

        er_l =[]
        if distance(data.T[R1[0]], data.T[R1[1]])  != distance(data.T[R2[0]], data.T[R2[1]]):
            print('ERROR-------------', i)
            print(f'test: {R1}, dist {distance(data.T[R1[0]], data.T[R1[1]])}')
            print(f'right: {R2}, dist {distance(data.T[R2[0]], data.T[R2[1]])}')
            print('--------------------')
            er_l.append(data)
        else:
            print('OK',i)
        return er_l



def angle1(con_pol, m,n):
    
    mod = con_pol.shape[0]

    k1 = np.arctan2((con_pol[(m+1) % mod][1] - con_pol[m% mod][1]),(con_pol[(m+1) % mod][0] - con_pol[m% mod][0]))
    
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    
    # k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    # k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    res = np.abs(k1 - k2)
    
    if k2< k1:
        # res = np.pi -  np.sign(k2)*k2 - k1
        res = 2*np.pi  - np.sign(k1)*k1  - np.sign(k2) * k2

    # res = np.abs(k1 - k2)

    print(f'k1 {k1}, k2 {k2}, res {res}')
    return np.abs(res)


def angle (con_pol, m,n):
    mod = con_pol.shape[0]
    # print(f'first line start {con_pol[(m)%mod]}, end {con_pol[(m+1)%mod]}')
    # print(f'second line start {con_pol[(n)%mod]}, end {con_pol[(n+1)%mod]}')
    k1 = np.arctan2((con_pol[(m+1) % mod][1] - con_pol[m% mod][1]),(con_pol[(m+1) % mod][0] - con_pol[m% mod][0]))
    
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    print(f'before rotate k1 {k1}, k2 {k2}')

    k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    print(f'after rotate k1 {k1}, k2 {k2}')
    
    if k2< k1:
        res = np.pi  - (k1 - k2)
    else:
        res = np.abs(k1 - k2)

    print(f'k1 {k1}, k2 {k2}, res {res}')
    return np.abs(res)



if __name__ == "__main__":

    
    lr = test()
    print(len(lr))


    # data = random_data(x_limits, y_limits,N)
    # index_list = graham_scan(data)

    # R1 = quick_max_distance(data)
    # R2 = max_distance(data)

    # print(f'test: {R1}, dist {distance(data.T[R1[0]], data.T[R1[1]])}')
    # print(f'right: {R2}, dist {distance(data.T[R2[0]], data.T[R2[1]])}')

    # print('!!!!')

    # fig, ax = plt.subplots()
    
    # ax.set_xlim(*x_limits)
    # ax.set_ylim(*y_limits)
    # ax.axis("equal")
    # ax.scatter(data[0],data[1])
    
    # print(data.T[index_list[-1]])
    # print(len(index_list))
    # for i in range(len(index_list)-1):
    #     ax.plot([data[0][index_list[i]], data[0][index_list[i+1]]], [data[1][index_list[i]], data[1][index_list[i+1]]], c="r")
    #     ax.scatter(data[0][index_list[i]], data[1][index_list[i]], c = 'r')

    # ax.plot([data[0][R1[0]], data[0][R1[1]]], [data[1][R1[0]], data[1][R1[1]]])
    # ax.plot([data[0][R2[0]], data[0][R2[1]]], [data[1][R2[0]], data[1][R2[1]]])
    # plt.show()




