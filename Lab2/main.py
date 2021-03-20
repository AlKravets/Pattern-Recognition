import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import time
from scipy.spatial import ConvexHull
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
N =10

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
    '''
    Поиск выпуклой оболочки. Алгоритм Грэхема
    data: numpy array shape= (2,N) (data[0] координаты иксов, data[1] координаты игреков)
    '''
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

def first_angle(con_pol, m,n):
    """
    Это костыль для rotating_calipers
    Нужен для вычисления первой пары точек
    считает угол между отрезками, что образованы вершинами выпуклого многоуголькика
    первый отрезок создается из 2 вершин с индексами m+1, m
    второй отрезок создается из 2 вершин с индексами n, n+1
    Совмещает точки m+1, n. И считает угол по направлению часовой стрелки? (возможны ошибки)
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

    # print(f'k1 {k1}, k2 {k2}, res {res}')
    return res 

def angle (con_pol, m,n):
    """
    Функция для rotating_calipers
    считает угол между прямыми, что образованы вершинами выпуклого многоуголькика
    первая прямая создается из 2 вершин с индексами m, m+1
    вторая прямая создается из 2 вершин с индексами n, n+1
    Номера берутся по модулю количества вершин многоугольника
    Угол - это угол от первой прямой ко второй по направлению ПРОТИВ часовой стрелки
    """

    mod = con_pol.shape[0]
    # print(f'first line start {con_pol[(m)%mod]}, end {con_pol[(m+1)%mod]}')
    # print(f'second line start {con_pol[(n)%mod]}, end {con_pol[(n+1)%mod]}')
    k1 = np.arctan2((con_pol[(m+1) % mod][1] - con_pol[m% mod][1]),(con_pol[(m+1) % mod][0] - con_pol[m% mod][0]))
    
    k2 = np.arctan2((con_pol[(n+1) % mod][1] - con_pol[n% mod][1]),(con_pol[(n+1) % mod][0] - con_pol[n% mod][0]))
    # print(f'before rotate k1 {k1}, k2 {k2}')

    k1 = k1 if np.abs(k1)< np.pi/2 else k1 - k1/np.abs(k1)*np.pi
    k2 = k2 if np.abs(k2)< np.pi/2 else k2 - k2/np.abs(k2)*np.pi

    # print(f'after rotate k1 {k1}, k2 {k2}')
    
    if k2< k1:
        res = np.pi  - (k1 - k2)
    else:
        res = np.abs(k1 - k2)

    # print(f'k1 {k1}, k2 {k2}, res {res}')
    return np.abs(res)

def rotating_calipers(con_pol):
    """
    Измененный Shamos's algorithm
    принимает точки, что являются вершинами выпуклой оболочки
    """
    n = len(con_pol)

    # i0 = np.argmin(con_pol, axis=0)[0]
    i = 0
    j = i+1

    while first_angle(con_pol, i, j)< np.pi:
        j+=1
    
    yield i, j

    current = i
    i+=1
    while j != n+1:
        
        # print(f'current {current}, i & angle:  {i, angle(con_pol,current, i)}, j & angle: {j, angle(con_pol,current, j)}')
        if angle(con_pol,current, i) == angle(con_pol,current,j):
            yield i, j
            yield i+1, j
            yield i, j+1
            

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

def quick_max_distance(data, Use_library = False):
    '''
    Функция использует выпуклую оболочку
    и rotating calipers
    data: numpy array shape= (2,N) (data[0] координаты иксов, data[1] координаты игреков)
    Можно использовать библиотечную функцию поиска выпуклой оболочки, это быстрее
    '''
    # index_list1 = graham_scan(data)
    if not Use_library:
        index_list = graham_scan(data)
    else:
        hull = ConvexHull(data.T)
        index_list = list(hull.vertices)
    # print(index_list1)
    # print(index_list)

    con_pol = data.T[index_list]
    mod = con_pol.shape[0]
    # print(mod)
    max_d = 0
    for i in rotating_calipers(con_pol):
        # print(i)
        i= (i[0]% mod, i[1]% mod)
        if max_d< distance(con_pol[i[0]], con_pol[i[1]]):
            max_d = distance(con_pol[i[0]], con_pol[i[1]])
            res = i
            # print('res=',res)
    return index_list[res[0]], index_list[res[1]]
    
def test():
    k = 500

    for i in range(k):
        data = random_data(x_limits, y_limits,100)
        

        R1 = quick_max_distance(data, Use_library=True)
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


class CellForDots:
    def __init__(self, x_lim, y_lim, index_dots, flag_list):
        '''
        x_lim = [x_min, x_max], y_lim = [y_min,y_max]
        index_dots - индексы точек, что попадают внутрь ячейки
        flag_list = numpy array (bool , bool, bool, bool)
        flag_list - массив их 4 bool, индикатор того, является ли стенка якейки внешней для всего множества точек.
        значения соответствуют x_min, x_max, y_min,y_max соответственно 
        '''
        
        self.x_lim = x_lim
        self.y_lim = y_lim
        
        self.flag_list = flag_list

        self.index_dots = index_dots

        self.division = True if len(self.index_dots) > 1 else False

    def data_for_plot(self):
        res_x = [
            self.x_lim[0],
            self.x_lim[0],
            self.x_lim[1],
            self.x_lim[1],
            self.x_lim[0],
        ]
        res_y = [
            self.y_lim[0],
            self.y_lim[1],
            self.y_lim[1],
            self.y_lim[0],
            self.y_lim[0],
        ]
    
        return res_x, res_y

def cell_division(all_dots, cell):
    '''
    Деление одной клетки на 2 новые. Если клетка не делится (внутри только 1 точка), то вернет ту же клетку
    если новая клетка не содержит стороны, что является частью внешней границы точек, то функция ее уничтожит.
    all_dots - массив точек, это numpy массив размера (N,2)
    '''
    res = []

    if not cell.division:
        return [cell]

    if cell.x_lim[1] - cell.x_lim[0] < cell.y_lim[1] - cell.y_lim[0]:
        divider = (cell.y_lim[1] - cell.y_lim[0])/2 + cell.y_lim[0]
        index_divider = 1
    else:
        divider = (cell.x_lim[1] - cell.x_lim[0])/2 + cell.x_lim[0]
        index_divider = 0

    b_arr = np.where( (all_dots[cell.index_dots]).T[index_divider] < divider, True, False)

    in_dots_less = cell.index_dots[b_arr]
    
    in_dots_greater = cell.index_dots[np.logical_not(b_arr)]

    flag_list_less = cell.flag_list.copy()
    flag_list_less[index_divider*2+1] = False

    flag_list_greater = cell.flag_list.copy()
    flag_list_greater[index_divider*2] = False


    x_lim_less = cell.x_lim[:]
    x_lim_greater = cell.x_lim[:]

    y_lim_less = cell.y_lim[:]
    y_lim_greater = cell.y_lim[:]

    if index_divider==1:
        y_lim_less[1] = divider
        y_lim_greater[0] = divider
    else:
        x_lim_less[1] = divider
        x_lim_greater[0] = divider

    
    cell_less = CellForDots(x_lim_less, y_lim_less, in_dots_less, flag_list_less)
    # print(f'less flag_list_less {flag_list_less}, division {cell_less.division}')
    if cell_less.index_dots.shape[0] >0 and np.sum(flag_list_less):
        res.append(cell_less)
    
    cell_greater = CellForDots(x_lim_greater, y_lim_greater, in_dots_greater, flag_list_greater)
    # print(f'greater flag_list_greater {flag_list_greater}, division {cell_greater.division}')
    if cell_greater.index_dots.shape[0] >0 and np.sum(flag_list_greater):
        res.append(cell_greater)

    return res


def create_cell_list(data, max_level_division= 6):
    '''
    Функция делит область с точками на клетки CellForDots
    data: numpy array shape= (2,N) (data[0] координаты иксов, data[1] координаты игреков)
    max_level_division - количество уровней разделений области
    '''

    all_dots =data.T

    cell_list = []

    x_lim = [np.min(all_dots.T[0]), np.max(all_dots.T[0])]
    y_lim = [np.min(all_dots.T[1]), np.max(all_dots.T[1])]
    
    index_dots = np.arange(all_dots.shape[0])

    flag_list = np.array([True]*4)

    cell_list.append(CellForDots(x_lim,y_lim,index_dots, flag_list))

    for i in range(max_level_division):
        n = len(cell_list)

        for j in range(n):
            cell_list += cell_division(all_dots, cell_list.pop(0))
            # print(f'i= {i}, n={n}, len(cell_list) ={len(cell_list)}')
        
    # for cell in cell_list:
    #     plt.plot(cell.data_for_plot()[0], cell.data_for_plot()[1], c='red')
    #     plt.scatter(data[0][cell.index_dots], data[1][cell.index_dots], marker='x')    

    return cell_list


def distance_in_cell_list(data,cell_list):
    '''
    Служебная функция для max_distance_by_cells
    data: numpy array shape= (2,N) (data[0] координаты иксов, data[1] координаты игреков)
    cell_list - список клеток на которые разделена область
    '''
    all_dots = data.T
        
    index_list = []

    for cell in cell_list:
        index_list += list(cell.index_dots)

    # print(index_list)

    data_in_cell_list = all_dots[index_list]

    max =0
    index = []
    for in1, dot1 in enumerate(data_in_cell_list):
        for in2,dot2 in enumerate(data_in_cell_list[in1:], start=in1):
            if max < distance(dot1, dot2):
                max = distance(dot1, dot2)
                index = [in1,in2]
                # print(index)
    return index_list[index[0]], index_list[index[1]]

def max_distance_by_cells(data,max_level_division= 10):
    '''
    Поиск диаметра множества использующий деление области на клетки
    data: numpy array shape= (2,N) (data[0] координаты иксов, data[1] координаты игреков)
    max_level_division - количество уровней разделений области
    Возвращает индексы точек между которыми можно провести диаметр
    '''
    cell_list = create_cell_list(data, max_level_division= max_level_division)
    return distance_in_cell_list(data, cell_list)


def time_table(functions_list, special_params_list, size_list, delimiter = ' & ', decimal_places= 4,x_limits = x_limits, y_limits =y_limits):
    '''
    Функция для создания таблицы времени работы алгоритмов. принимает список функций,
    список словарей дополнительных параметров к ним. Если параметров нет, то передается пустой словарь.
    size_list - список с количествами точек. 
    '''

    assert len(functions_list)==len(special_params_list)

    time_list = []

    for size in size_list:
        data = random_data(x_limits, y_limits,size)
        time_list.append([])

        for i, func in enumerate(functions_list):
            t0 = time.time()
            func(data, **special_params_list[i])
            time_list[-1].append(time.time()-t0)
    
    for j in range(len(functions_list)):
        for tm  in time_list:
            print(delimiter, round(tm[j],decimal_places),sep='', end='')
        print()    
    

def plots_for_quick_distance(x_limits = x_limits, y_limits = y_limits, N = N, title = 'Convex hull', nsize =0.4):
    '''
    Иллюстрация для алгоритма с выпуклой оболочкой
    '''
    data = random_data(x_limits, y_limits,N)
    index_list = graham_scan(data)
    R = quick_max_distance(data)

    fig, ax = plt.subplots(figsize =(nsize* (x_limits[1]- x_limits[0]),nsize* (y_limits[1]- y_limits[0])))   
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.axis("equal")
    ax.scatter(data[0],data[1], color = 'b', s= 2)
    
    mod = len(index_list)

    for i in range(len(index_list)):
        ax.plot([data[0][index_list[i]], data[0][index_list[(i+1)% mod]]], [data[1][index_list[i]], data[1][index_list[(i+1)%mod]]], c="r")
        ax.scatter(data[0][index_list[i]], data[1][index_list[i]], c = 'r')

    ax.plot([data[0][R[0]], data[0][R[1]]], [data[1][R[0]], data[1][R[1]]], c = 'y')
    ax.grid(alpha = 0.2)
    ax.set_title(title)
    plt.show()

def plots_for_cells_distance(x_limits = x_limits, y_limits = y_limits, N = N, title = 'Bounding box', nsize =0.4, max_level_division = 10):
    '''
    Иллюстрация для алгоритма с клетками
    '''
    data = random_data(x_limits, y_limits,N)
    cell_list = create_cell_list(data,max_level_division =max_level_division)
    R = max_distance_by_cells(data, max_level_division =max_level_division)

    fig, ax = plt.subplots(figsize =(nsize* (x_limits[1]- x_limits[0]),nsize* (y_limits[1]- y_limits[0])))   
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.axis("equal")
    ax.scatter(data[0],data[1], color = 'b', s = 2)
    
    for cell in cell_list:
        plt.plot(cell.data_for_plot()[0], cell.data_for_plot()[1], c='red')
        plt.scatter(data[0][cell.index_dots], data[1][cell.index_dots], marker='x', c = 'b')


    ax.plot([data[0][R[0]], data[0][R[1]]], [data[1][R[0]], data[1][R[1]]], c = 'y')
    ax.grid(alpha = 0.2)
    ax.set_title(title)
    plt.show()


def spc_plot_for_cells_distance():
    '''
    Последовательная иллюстрация для алгоритма с клетками
    '''

    x_limits = [-10,10]
    y_limits =[-5,5]
    N = 6
    max_level_division =5
    data = random_data(x_limits, y_limits,N)
    # cell_list = create_cell_list(data max_level_division= max_level_division)
    
    
    R = max_distance_by_cells(data,max_level_division=max_level_division )

    fig, ax = plt.subplots(2,2,figsize =(15,15))
    ax = ax.ravel()
    for lv in range(1,max_level_division):
        i = lv-1   
        ax[i].set_xlim(*x_limits)
        ax[i].set_ylim(*y_limits)
        ax[i].axis("equal")
        ax[i].scatter(data[0],data[1], color = 'b', s= 2)
        
        cell_list= create_cell_list(data, max_level_division= lv)
        for cell in cell_list:
            ax[i].plot(cell.data_for_plot()[0], cell.data_for_plot()[1], c='red')
            ax[i].scatter(data[0][cell.index_dots], data[1][cell.index_dots], marker='x', c = 'b')


        ax[i].plot([data[0][R[0]], data[0][R[1]]], [data[1][R[0]], data[1][R[1]]], c = 'y')
        ax[i].grid(alpha = 0.2)
        title = f'made divisions: ${lv}$'
        ax[i].set_title(title)
    plt.show()





if __name__ == "__main__":

    
    functions_list = [
        max_distance,
        quick_max_distance,
        max_distance_by_cells]
    params = [
        {},
        {'Use_library' : False},
        {'max_level_division':10}
    ]

    sizes = [10,100, 500, 1000]
    time_table(functions_list, params, size_list= sizes)

    plots_for_quick_distance(N = 50)
    plots_for_cells_distance(N = 100,max_level_division=10)

    spc_plot_for_cells_distance()




