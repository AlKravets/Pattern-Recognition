import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
# np.random.seed(20)
# Получение данных

x_limits = [-10,10]
y_limits = [-5,5]
N = 15

def random_data(x_limits, y_limits,N):
    """
    Функция создает массив точек на плоскости в заданных ограничениях 
    с равномерным распределением
    ограничения (x_limits, y_limits) заданы в виде (x_min, x_max)
    Размер выходного массива (2,N)
    """
    return np.vstack((np.random.uniform(x_limits[0],x_limits[1], N),
            np.random.uniform(y_limits[0],y_limits[1], N)))

def text_data(filename):
    pass
    # TODO

data = random_data(x_limits, y_limits, N)
fig_list = []

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

fig_list.append(first_pic(data, x_limits, y_limits, title= "Полученный набор точек"))


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

index_max = max_distance(data)


print(index_max)
print(data[:, index_max[0]], data[:, index_max[1]])

def second_pic(data, x_limits, y_limits, index_max,  size =7, title = ""):
    """
    Рисует полученные точки и отрезок, котрый соединяет самые отдаленные точки
    """
    fig= first_pic(data, x_limits, y_limits, size , title)
    ax = fig.axes[0]
    ax.plot([data[0,index_max[0]], data[0,index_max[1]]], [data[1,index_max[0]], data[1,index_max[1]]])
    return fig

fig_list.append(second_pic(data, x_limits, y_limits, index_max, title= "Максимально отдаленные точки"))

def mass_centr(data):
    return np.sum(data, axis =1)/ data.shape[1]

centr = mass_centr(data)

print(centr)

def rotate_point (point, center, angle):
    """
    Поворот точек point (хранятся в виде (2,N)) относительно center на угол angle
    angle в радианах и angle лежит в (-pi/2 , pi/2)
    """
    ox = (point[0] - center[0])*np.cos(angle) - (point[1]- center[1])*np.sin(angle) + center[0]
    oy = (point[0] - center[0])*np.sin(angle) + (point[1]- center[1])*np.cos(angle) + center[1]
    return np.vstack((ox,oy))

def find_angle(data, index_max):
    """
    Функция ищет угол наклона [-pi/2, pi/2] отрезка порожденного индексами index_max
    """
    return np.arctan((data[1,index_max[1]] -data[1,index_max[0]])/ (data[0,index_max[1]] -data[0,index_max[0]]))

angle = find_angle(data, index_max)
# умножаем угол на -1
rdata = rotate_point(data, data[:,index_max[0]], -1*angle)

def test_third_pic(rdata, index_max, title = ""):
    """
    Рисует точки и отрезок после поворота
    """
#     print(rdata)
    fig, ax = plt.subplots()
    ax.scatter(rdata[0],rdata[1])
    ax.plot([rdata[0,index_max[0]], rdata[0,index_max[1]]], [rdata[1,index_max[0]], rdata[1,index_max[1]]])
    plt.show()

def vertical_segments(rdata):
    """
    Функция находит индексы точек с имнимальным и максимальным значениями по оси y 
    """
    return [np.argmin(rdata,axis=1)[1], np.argmax(rdata,axis=1)[1]]

v_index = vertical_segments(rdata)
print(v_index)

[rdata[:,i] for i in v_index]

def vertical_perpendiculars(rdata, index_max, v_index):
    """
    Функция возвращает 2 отрезка, перпендикуляры максимальной длины, к горизонтальному отрезку,
    который задан индексами index_max
    """
    seg1 = np.array([rdata.T[v_index[0]], [rdata.T[v_index[0],0], rdata.T[index_max[0],1]]]).T
    seg2 = np.array([rdata.T[v_index[1]], [rdata.T[v_index[1],0], rdata.T[index_max[0],1]]]).T
    return seg1, seg2

segments = np.array(vertical_perpendiculars(rdata, index_max, v_index))

def third_pic(rdata, index_max,v_index, segments, size = 7, title= ""):
    """
    Повернутый график с максимальными вертикальными отрезками
    """
    new_y_limits = rdata[1,v_index[0]]*1.1, rdata[1,v_index[1]]*1.1
    new_x_limits = rdata[0,index_max[0]]*1.1, rdata[0,index_max[1]]*1.1
    print(f"NEw lim y: {new_y_limits}, x: {new_x_limits}")
    fig = second_pic(rdata, new_x_limits, new_y_limits,index_max, size, title )
    ax = fig.axes[0]
    ax.plot(segments[0][0], segments[0][1], color ='red')
    ax.plot(segments[1][0], segments[1][1], color ='red')
    return fig


fig_list.append(third_pic(rdata,index_max,v_index, segments,title="Повернутый график с максимальными вертикальными отрезками"))

def compression_ratio(rdata, index_max, v_index):
    """
    находит коэффициент сжатия/растяжения по x для получения квадрата,
    а именно равенства отрезков, что были получены из index_max, v_index
    """
    x_length = np.abs(rdata[0,index_max[1]] - rdata[0,index_max[0]])
    y_length = np.abs(rdata[1,v_index[1]] - rdata[1,v_index[0]])
    return y_length/ x_length

koef = compression_ratio(rdata, index_max, v_index)


def make_square_data(rdata, segments, koef):
    """
    Функция изменяет данные, сжимает или растягивает по координате x для получения квадрата,
    а именно равенства отрезков, что были получены из index_max, v_index
    """
    s_rdata = rdata.copy()
    s_rdata[0]= s_rdata[0]* koef
    s_segments= segments.copy()
    for i in range(len(s_segments)):
        s_segments[i][0]*= koef
    return s_rdata, s_segments

s_rdata, s_segments = make_square_data(rdata, segments, koef)

fig_list.append(third_pic(s_rdata,index_max,v_index, s_segments,title="Квадратный график"))

class MyCircle:
    """
    класс круга, который хранит количество точек, что попали внутрь (включая границу). Для этого передается data
    """
    def __init__(self, xy, radius, data):
        self.xy = xy
        self.radius = radius
        self.count = self._count_of_inside_dots(data)
        # print(self.count)
        
    def _count_of_inside_dots(self,data):
        """
        функция делает перефор точек data и возвращает кол-во попавших внутрь окружности (включая границу)
        """
        # транспонируем для удобства вычисления расстояния
        count =0
        for dot in data.T:
            if distance(dot, self.xy) <= self.radius:
                count+=1
        return count

    def __str__(self):
        return f"MyCircle xy: {self.xy}, radius: {self.radius},count: {self.count}"


##
mycircle_list = []
new_centr = mass_centr(s_rdata)
for dot in s_rdata.T:
    # print(f"{distance(dot, new_centr)}, dot {dot}")
    mycircle_list.append(MyCircle(new_centr, distance(dot, new_centr), s_rdata))
##
print(*mycircle_list, sep='\n')

def fourth_pic(s_rdata, index_max,v_index, segments, centr, mycircle_list,size = 7, title= ""):
    """
    Рисунок для кваратной области с концентрическими окружностями
    """
    fig = third_pic(s_rdata, index_max, v_index, segments, size = size, title = title)
    ax = fig.axes[0]
    ax.scatter(centr[0], centr[1], color = "y")
    for circle in mycircle_list:
        ax.add_patch(ptc.Circle(circle.xy,circle.radius, fill= False, color ="green"))
    return fig

fig_list.append(fourth_pic(s_rdata,index_max,v_index, s_segments, new_centr, mycircle_list, title="Квадратный график с окружностями"))


class MyEllipse:
    """
    Класс эллипсов с кол-вом точек, что попали внутрь (включая границу)
    """
    def __init__(self, xy, width, height, angle=0, count=0):
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = angle
        self.count = count
    def __str__(self):
        return f"MyEllipse xy: {self.xy}, width: {self.width}, height: {self.height}, angle: {self.angle}, count: {self.count}"

##
myellipse_list = []
r_centr  = mass_centr(rdata)
inv_koef = 1/koef
for circle in mycircle_list:
    myellipse_list.append(MyEllipse(r_centr, 2*circle.radius*inv_koef, 2*circle.radius, angle=0, count=circle.count))

print(*myellipse_list, sep="\n")

def fifth_pic(rdata, index_max, v_index, segments, centr, myellipse_list, size =7, title =""):
    """
    Рисует эллипся на повернутом рисунке
    """
    fig = third_pic(rdata, index_max, v_index, segments, size = size, title = title)
    ax = fig.axes[0]
    ax.scatter(centr[0], centr[1], color = "y")
    for ell in myellipse_list:
        ax.add_patch(ptc.Ellipse(ell.xy, ell.width, ell.height, angle = ell.angle, fill= False, color ="green"))
    return fig

fig_list.append(fifth_pic(rdata, index_max, v_index, segments, r_centr, myellipse_list, title = "Повернутый рисунок с эллипсами"))

##
centr = rotate_point(r_centr,data[:,index_max[0]], angle)
for ell in myellipse_list:
    ell.angle = np.degrees(angle)
    ell.xy = centr


print(*myellipse_list, sep="\n")

segments_r = []
for seg in segments:
    segments_r.append(rotate_point(seg,data[:,index_max[0]], angle))

fig_list.append(fifth_pic(data, index_max, v_index, segments_r, centr, myellipse_list, title = "рисунок с эллипсами"))



if __name__ =="__main__":
    # data = random_data(x_limits, y_limits, N)
    # print(data.shape)
    # print(data[:,1])
    plt.show()
    # pass

    # fig_list = []

    # fig1, ax1 = plt.subplots()
    # ax1.set(xlim=(-10,10), ylim= (-10, 10))
    # ell= ptc.Ellipse((0,0), 6, 2,angle= np.degrees(np.pi/4))
    # ax1.add_patch(ell)

    # plt.show()