import numpy as np
import math
# from numpy import linalg


# 3d bounding box
class BBox3D:
    def set_by_object(self, object):
        xmin = object[:, 0].min()
        xmax = object[:, 0].max()
        ymin = object[:, 1].min()
        ymax = object[:, 1].max()
        zmin = object[:, 2].min()
        zmax = object[:, 2].max()
        self.min = np.array([xmin, ymin, zmin])
        self.max = np.array([xmax, ymax, zmax])

    def set_by_bbox(self,center,size):
        self.min = np.empty((3))
        self.max = np.empty((3))
        for i in range(3):
            self.min[i] = center[i] - (size[i]/2)
            self.max[i] = center[i] + (size[i]/2)

    def __repr__(self):
        return f'({self.min[0],self.min[1],self.min[2]}),({self.max[0],self.max[1],self.max[2]})'

    def get_length(self,axis):
        if axis == 0:
            return (self.max[0] - self.min[0])
        elif axis == 1:
            return (self.max[1] - self.min[1])
        elif axis == 2:
            return (self.max[2] - self.min[2])

    def get_size(self):
        x = self.get_length(0)
        y = self.get_length(1)
        z = self.get_length(2)
        size = np.array([x,y,z])
        return size

    def get_volume(self):
        x = self.get_length(0)
        y = self.get_length(1)
        z = self.get_length(2)
        volume = x*y*z
        return volume

    def get_CenterPoint(self):
        x = (self.max[0] + self.min[0])/2
        y = (self.max[1] + self.min[1])/2
        z = (self.max[2] + self.min[2])/2
        point = np.array([x,y,z])
        return point

    def get_loc_size(self):
        point = self.get_CenterPoint()
        size = self.get_size()
        affine = np.concatenate((point, size), axis=0)
        return affine

    def corner(self):
        corners = np.array([[
            [self.min[0],self.max[0],self.max[0],self.min[0],self.min[0],self.max[0],self.max[0],self.min[0]],
            [self.min[1],self.min[1],self.max[1],self.max[1],self.min[1],self.min[1],self.max[1],self.max[1]],
            [self.min[2],self.min[2],self.min[2],self.min[2],self.max[2],self.max[2],self.max[2],self.max[2]]
        ]])
        return corners

    def distance_id(self,distance_per):
        if distance_per <= 0.1:
            distance_idx = 0
        elif distance_per <= 0.4:
            distance_idx = 1
        else:
            distance_idx = 2
        return distance_idx

    def in_choose_direction(self,bbox1,bbox2):
        center_point = bbox2.get_CenterPoint()
        top = (bbox1.max[2] - center_point[2])
        down = (center_point[2] - bbox1.min[2])
        front = (bbox1.max[0] - center_point[0])
        back = (center_point[0] - bbox1.min[0])
        right = (bbox1.max[1] - center_point[1])
        left = (center_point[1] - bbox1.min[1])
        choose_direction = np.array([top, down, front, back, right, left])
        direction_idx = choose_direction.argmin()
        if direction_idx == 0 or direction_idx == 1:
            distance_per = (choose_direction[direction_idx] - bbox2.get_length(2) / 2)/bbox1.get_length(2)
        elif direction_idx == 2 or direction_idx == 3:
            distance_per = (choose_direction[direction_idx] - bbox2.get_length(0) / 2) / bbox1.get_length(0)
        else:
            distance_per = (choose_direction[direction_idx] - bbox2.get_length(1) / 2) / bbox1.get_length(1)
        # 0<distance_per<0.5
        distance_idx = self.distance_id(distance_per)
        return direction_idx,distance_idx

    def out_chhoose_direction(self,bbox,axis):
        distance_idx = None
        state = None
        direction = None
        if axis == 2:
            flag = self.is_UpDownSurfcae_overlap(bbox)
        elif axis ==1:
            flag = self.is_RightLeftSurface_overlap(bbox)
        elif axis ==0:
            flag = self.is_FrontBackSurface_overlap(bbox)
        dis1 = bbox.min[axis] - self.max[axis]
        dis2 = self.min[axis] - bbox.max[axis]
        flag1 = 0
        # 沿着x or y or z轴重叠但重叠长度<=0.05时，算不重叠
        if abs(dis1) < abs(dis2) and dis1 < 0:
            distance_per = abs(bbox.min[axis] - self.max[axis]) / bbox.get_length(axis)
            if distance_per <= 0.05:
                flag1 = 1
        if abs(dis1) > abs(dis2) and dis2 < 0:
            distance_per = abs(self.min[axis] - bbox.max[axis]) / bbox.get_length(axis)
            if distance_per <= 0.05:
                flag1 = 2


        if self.max[axis] <= bbox.min[axis] and flag == 0:
            distance_per = (bbox.min[axis] - self.max[axis]) / bbox.get_length(axis)
            state = Space_Relationship.state.index('Object_Out')
            if axis == 2:
                direction = Space_Relationship.Direction.index('Bottom')
            elif axis == 1:
                direction = Space_Relationship.Direction.index('Left')
            elif axis == 0:
                direction = Space_Relationship.Direction.index('Back')
            distance_idx = self.distance_id(distance_per)
        elif flag1 == 1 and flag == 0:
            if axis == 2:
                direction = Space_Relationship.Direction.index('Bottom')
            elif axis == 1:
                direction = Space_Relationship.Direction.index('Left')
            elif axis == 0:
                direction = Space_Relationship.Direction.index('Back')
            state = Space_Relationship.state.index('Object_Out')
            distance_idx = 0



        if self.min[axis] >= bbox.max[axis] and flag == 0:
            distance_per = (self.min[axis] - bbox.max[axis]) / bbox.get_length(axis)
            state = Space_Relationship.state.index('Object_Out')
            if axis == 2:
                direction = Space_Relationship.Direction.index('Top')
            elif axis == 1:
                direction = Space_Relationship.Direction.index('Right')
            elif axis == 0:
                direction = Space_Relationship.Direction.index('Front')
            distance_idx = self.distance_id(distance_per)
        elif flag1 == 2 and flag == 0:
            if axis == 2:
                direction = Space_Relationship.Direction.index('Top')
            elif axis == 1:
                direction = Space_Relationship.Direction.index('Right')
            elif axis == 0:
                direction = Space_Relationship.Direction.index('Front')
            state = Space_Relationship.state.index('Object_Out')
            distance_idx = 0

        return direction,state,distance_idx

    def Intersect_choose_direction(self,bbox):
        distance_idx = None
        state = None
        direction = None
        x_max,x_min = self.cross_length(bbox,0)
        y_max,y_min = self.cross_length(bbox,1)
        z_max,z_min = self.cross_length(bbox,2)
        x_length = x_max - x_min
        y_length = y_max - y_min
        z_length = z_max -z_min
        xy = x_length * y_length
        yz = y_length * z_length
        xz = x_length * z_length
        front_v = yz * abs(self.max[0] - x_min)
        back_v = yz * abs(self.min[0] - x_max)
        top_v = xy * abs(self.max[2] - z_min)
        bottom_y = xy * abs(self.min[2] - z_max)
        left_v = xz * abs(self.min[1] - y_max)
        right_v = xz * abs(self.max[1] - y_min)
        v_all = np.array((top_v,bottom_y,front_v,back_v,right_v,left_v))
        center_point = self.get_CenterPoint()
        # 没有体积，例如sink的counter,有一个维度为0
        if v_all.mean() == 0.0:
            state = Space_Relationship.state.index('Intersect_in')
            if xy != 0.0:
                dis1 = abs(center_point[2] - bbox.max[2])
                dis2 = abs(center_point[2] - bbox.min[2])
                length = bbox.max[2] - bbox.min[2]
                if dis1 < dis2:
                    direction = 0
                    distance_per = dis1 / length
                    distance_idx = self.distance_id(distance_per)
                else:
                    direction = 1
                    distance_per = dis2 / length
                    distance_idx = self.distance_id(distance_per)
            elif yz !=0.0:
                dis1 = abs(center_point[0] - bbox.max[0])
                dis2 = abs(center_point[0] - bbox.min[0])
                length = bbox.max[0] - bbox.min[0]
                if dis1 < dis2:
                    direction = 2
                    distance_per = dis1 / length
                    distance_idx = self.distance_id(distance_per)
                else:
                    direction = 3
                    distance_per = dis2 / length
                    distance_idx = self.distance_id(distance_per)
            elif xz != 0.0:
                dis1 = abs(center_point[1] - bbox.max[1])
                dis2 = abs(center_point[1] - bbox.min[1])
                length = bbox.max[1] - bbox.min[1]
                if dis1 < dis2:
                    direction = 4
                    distance_per = dis1 / length
                    distance_idx = self.distance_id(distance_per)
                else:
                    direction = 5
                    distance_per = dis2 / length
                    distance_idx = self.distance_id(distance_per)
            return direction, state, distance_idx
        # 有体积
        direction = v_all.argmax()
        if direction == 0:
            distance_rough = center_point[2] - bbox.max[2]
            distance_per = abs(distance_rough) / (self.max[2] - self.min[2])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_in')
            else:
                state = Space_Relationship.state.index('Intersect_out')

            distance_idx = self.distance_id(distance_per)
        elif direction == 1:
            distance_rough = center_point[2] - bbox.min[2]
            distance_per = abs(distance_rough) / (self.max[2] - self.min[2])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_out')
            else:
                state = Space_Relationship.state.index('Intersect_in')

            distance_idx = self.distance_id(distance_per)
        elif direction == 2:
            distance_rough = center_point[0] - bbox.max[0]
            distance_per = abs(distance_rough) / (self.max[0] - self.min[0])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_in')
            else:
                state = Space_Relationship.state.index('Intersect_out')

            distance_idx = self.distance_id(distance_per)
        elif direction == 3:
            distance_rough = center_point[0] - bbox.min[0]
            distance_per = abs(distance_rough) / (self.max[0] - self.min[0])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_out')
            else:
                state = Space_Relationship.state.index('Intersect_in')

            distance_idx = self.distance_id(distance_per)
        elif direction == 4:
            distance_rough = center_point[1] - bbox.max[1]
            distance_per = abs(distance_rough) / (self.max[1] - self.min[1])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_in')
            else:
                state = Space_Relationship.state.index('Intersect_out')

            distance_idx = self.distance_id(distance_per)
        elif direction == 5:
            distance_rough = center_point[1] - bbox.min[1]
            distance_per = abs(distance_rough) / (self.max[1] - self.min[1])
            if distance_rough <= 0:
                state = Space_Relationship.state.index('Intersect_out')
            else:
                state = Space_Relationship.state.index('Intersect_in')

            distance_idx = self.distance_id(distance_per)

        return direction,state,distance_idx

    # self的面与bbox的面对应是否重叠
    def is_UpDownSurfcae_overlap(self,bbox):
        flag = 0
        if self.min[0] > bbox.max[0] or self.max[0] < bbox.min[0] \
            or self.min[1] > bbox.max[1] or self.max[1] < bbox.min[1]:
            flag = 1
        return flag
    def is_RightLeftSurface_overlap(self,bbox):
        flag = 0
        if self.min[2] > bbox.max[2] or self.max[2] < bbox.min[2] \
            or self.min[0] > bbox.max[0] or self.max[0] < bbox.min[0]:
            flag = 1
        return flag
    def is_FrontBackSurface_overlap(self,bbox):
        flag = 0
        if self.min[1] > bbox.max[1] or self.max[1] < bbox.min[1] \
            or self.min[2] > bbox.max[2] or self.max[2] < bbox.min[2]:
            flag = 1
        return flag

    def isCollided(self,bbox):
        min = bbox.min
        max = bbox.max
        if(self.min[0] >= max[0] or self.max[0] <= min[0]):
            return False
        if(self.min[1] >= max[1] or self.max[1] <= min[1]):
            return False
        if(self.min[2] >= max[2] or self.max[2] <= min[2]):
            return False
        return True

    @property
    def volume(self):
        dis = self.max - self.min
        return dis[0]*dis[1]*dis[2]

    def cross_length(self,bbox,axis):
        min = self.min[axis]
        if self.min[axis]<bbox.min[axis]:
            min = bbox.min[axis]
        max = self.max[axis]
        if self.max[axis]>bbox.max[axis]:
            max = bbox.max[axis]

        return max,min

    def is_add_edge(self,bbox):
        # object in or surround
        x_minmax = sorted([self.min[0], self.max[0], bbox.min[0], bbox.max[0]])
        y_minmax = sorted([self.min[1], self.max[1], bbox.min[1], bbox.max[1]])
        z_minmax = sorted([self.min[2], self.max[2], bbox.min[2], bbox.max[2]])
        x_length = x_minmax[2] - x_minmax[1]
        y_length = y_minmax[2] - y_minmax[1]
        z_length = z_minmax[2] - z_minmax[1]
        threshold = 0.2
        if self.is_UpDownSurfcae_overlap(bbox) == 0:
            surface_per1 = (x_length*y_length) / (self.get_length(0)*self.get_length(1))
            surface_per2 = (x_length * y_length) / (bbox.get_length(0) * bbox.get_length(1))
            if surface_per1 >= threshold or surface_per2 >= threshold:
                return True
        if self.is_FrontBackSurface_overlap(bbox) == 0:
            surface_per1 = (y_length * z_length) / (self.get_length(1) * self.get_length(2))
            surface_per2 = (y_length * z_length) / (bbox.get_length(1) * bbox.get_length(2))
            # print(surface_per)
            if surface_per1 >= threshold or surface_per2 >= threshold:
                return True
        if self.is_RightLeftSurface_overlap(bbox) == 0:
            surface_per1 = (x_length * z_length) / (self.get_length(0) * self.get_length(2))
            surface_per2 = (x_length * z_length) / (bbox.get_length(0) * bbox.get_length(2))
            # print(surface_per)
            if surface_per1 >= threshold or surface_per2 >= threshold:
                return True
        return False

    # 添加垂直方向
    def vertical_direction(self, direction, bbox):
        x_length = bbox.get_length(0) / 3
        y_length = bbox.get_length(1) / 3
        z_length = bbox.get_length(2) / 3
        point = self.get_CenterPoint()
        center = bbox.get_CenterPoint()
        if Space_Relationship.Direction[direction] == 'Top' or Space_Relationship.Direction[direction] == 'Bottom':
            point1, point2 = point[0], point[1]
            center1, center2 = center[0], center[1]
            v_direction = self.choose_v_direction(point1, point2, center1, center2, x_length, y_length)
        elif Space_Relationship.Direction[direction] == 'Left' or Space_Relationship.Direction[direction] == 'Right':
            point1, point2 = point[0], point[2]
            center1, center2 = center[0], center[2]
            v_direction = self.choose_v_direction(point1, point2, center1, center2, x_length, z_length)
        else:
            point1, point2 = point[1], point[2]
            center1, center2 = center[1], center[2]
            v_direction = self.choose_v_direction(point1, point2, center1, center2, y_length, z_length)
        return v_direction

    def choose_v_direction(self, point1, point2, center1, center2, length1, length2):
        if point1 < (center1 - (length1 / 2)):
            if point2 < (center2 - (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('left-above')
            elif point2 > (center2 + (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('right-above')
            else:
                v_direction = Space_Relationship.Vertical_Direction.index('above')
        elif point1 > (center1 + (length1 / 2)):
            if point2 < (center2 - (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('left-below')
            elif point2 > (center2 + (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('right-below')
            else:
                v_direction = Space_Relationship.Vertical_Direction.index('below')
        else:
            if point2 < (center2 - (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('left')
            elif point2 > (center2 + (length2 / 2)):
                v_direction = Space_Relationship.Vertical_Direction.index('right')
            else:
                v_direction = Space_Relationship.Vertical_Direction.index('center')
        return v_direction

    def relation_to(self,bbox):
        # bbox 8个点都在内
        if (self.min >= bbox.min).all() and (self.max <= bbox.max).all():
            state = Space_Relationship.state.index('Object_In')
            direction, distance = self.in_choose_direction(bbox,self)
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                        Space_Relationship.Direction[direction],
            #                                        Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction

        if (self.min <= bbox.min).all() and (self.max >= bbox.max).all():
            state = Space_Relationship.state.index('Surround')
            direction, distance = self.in_choose_direction(self,bbox)
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                        Space_Relationship.Direction[direction],
            #                                        Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction

        # 8个点在外,2表示z轴，1表示y轴，0表示x轴
        direction,state,distance = self.out_chhoose_direction(bbox,2)
        if distance != None and state!=None and direction!=None:
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                                 Space_Relationship.Direction[direction],
            #                                                 Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction
        direction, state, distance = self.out_chhoose_direction(bbox, 1)
        if distance != None and state != None and direction != None:
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                                 Space_Relationship.Direction[direction],
            #                                                 Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction
        direction, state, distance = self.out_chhoose_direction(bbox, 0)
        if distance != None and state != None and direction != None:
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                                 Space_Relationship.Direction[direction],
            #                                                 Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction

        # bbox交叉
        direction, state, distance = self.Intersect_choose_direction(bbox)
        if distance != None and state != None and direction != None:
            # print("space relation: {} --- {} --- {}".format(Space_Relationship.state[state],
            #                                                 Space_Relationship.Direction[direction],
            #                                                 Space_Relationship.Distance[distance]))
            v_direction = self.vertical_direction(direction, bbox)
            # print("v_direction: {}".format(Space_Relationship.Vertical_Direction[v_direction]))
            return state, direction, distance, v_direction

        # return False

class Space_Relationship():
    state = ['Surround', 'Object_In', 'Intersect_in', 'Intersect_out', 'Object_Out']
    Direction = ['Top', 'Bottom', 'Front', 'Back', 'Right', 'Left']
    Distance = ['adjacent', 'proximal', 'distant']
    Vertical_Direction = ['center', 'above', 'below', 'left', 'right',
                          'left-above', 'right-above', 'left-below', 'right-below']
    st_num = len(state)
    dr_num = len(Direction)
    ds_num = len(Distance)
    vdr_num = len(Vertical_Direction)
    number = len(state) + len(Distance) + len(Direction) + len(Vertical_Direction)

class Obj_Interaction():
    function = ['Carrying', 'Contained', 'Hanging', 'Holding', 'Hung',
                'Infront', 'Lighted', 'Lying', 'Onside', 'Overhanging',
                'Pushing', 'Riding', 'Nailed', 'Sitting', 'Typing',
                'Surrounding', 'Supporting', 'Supported']
    number = len(function)

class Center_Node():
    category = ['Backpack','Basket','Bathtub','Bed','Bench',
            'Bicycle','Bowl','Chair','Cup','Desk',
            'DryingRack','Handcart','Hanger','Hook','Lamp',
            'Laptop','Shelf','Sink','Sofa','Stand',
            'Stool','Stroller','Table','Tvbench','Vase']
    number = len(category)

class Surround_Node():
    category = ['Human','Hand','Bottle','Banana','Apple','Egg','Food','Candy','Ground','Water',
            'Keyboard','Mat','Mouse','Monitor','Loudspeaker','Headphone','Chair','iPod','Case','Modem',
            'Box','Disk','Camera','Cabinet','Lamp','Note','Pen','Basket','Book','Plant',
            'Cup','Plate','Laptop','Bowl','Calendar','Telephone','Decoration','Mirror','Printer','Gamepad',
            'Stapler','Tape','T-Shirt','Dress','Shirt','Hoodie','Vest','Jacket','Hanger','Skirt',
            'Sweater','Coat','Brief','Hoddie','Jacker','Tie','Candle','Vase','Toy','Joystick',
            'Supplies','Suitcase','Carrot','Brick','Fruit','Hook','Bag','Wall','Towel','Umbrella',
            'Cap','Rope','Desk','Trophy','Photo','Ball','Hat','Clock','Kettle','Counter',
            'Tap','Baby','Flower','Orange','Knife','Strawberry','Fork','Tongs','U-disk','Cigarette',
            'Milk','Glass','TV','TapeDriver','DVD','VideoPlayer','Wii','Controller','XBOX','Projector',
            'Router','Branch']
    number = len(category)

class All_Node():
    category = ['Backpack','Basket','Bathtub','Bed','Bench','Bicycle','Bowl','Chair','Cup','Desk',
            'DryingRack','Handcart','Hanger','Hook','Lamp','Laptop','Shelf','Sink','Sofa','Stand',
            'Stool','Stroller','Table','Tvbench','Vase',
            'Human', 'Hand', 'Bottle', 'Banana', 'Apple', 'Egg', 'Food', 'Candy','Ground', 'Water',
            'Keyboard', 'Mat', 'Mouse', 'Monitor', 'Loudspeaker', 'Headphone', 'iPod', 'Case', 'Modem',
            'Box', 'Disk', 'Camera', 'Cabinet', 'Note', 'Pen', 'Book', 'Plant',
            'Plate', 'Calendar', 'Telephone', 'Decoration', 'Mirror', 'Printer', 'Gamepad',
            'Stapler', 'Tape', 'T-Shirt', 'Dress', 'Shirt', 'Hoodie', 'Vest', 'Jacket', 'Skirt',
            'Sweater', 'Coat', 'Brief', 'Hoddie', 'Jacker', 'Tie', 'Candle', 'Toy', 'Joystick',
            'Supplies', 'Suitcase', 'Carrot', 'Brick', 'Fruit', 'Bag', 'Wall', 'Towel', 'Umbrella',
            'Cap', 'Rope', 'Trophy', 'Photo', 'Ball', 'Hat', 'Clock', 'Kettle', 'Counter',
            'Tap', 'Baby', 'Flower', 'Orange', 'Knife', 'Strawberry', 'Fork', 'Tongs', 'U-disk', 'Cigarette',
            'Milk', 'Glass', 'TV', 'TapeDriver', 'DVD', 'VideoPlayer', 'Wii', 'Controller', 'XBOX', 'Projector',
            'Router', 'Branch']
    number = len(category)