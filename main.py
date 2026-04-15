import random
import pygame as pg
import numpy as np

width, height = 1200, 900
margin = 50

drone_number = 10

task_spawn_interval = 5000
task_radius = 8

min_drone_distance = 150
drone_arm_length = 15
drone_rotor_protect_radius = 15
drone_speed = 1.5
drone_rotor_length = 10
drone_rotor_speed = 0.35
camera_fov_angle = np.radians(60)
camera_view_distance = 200
drone_communication_radius = 200

background_color = (18, 20, 24)
drone_body_color = (200, 200, 200)
drone_rotor_color = (120, 200, 255)
drone_search_task_color = (2, 222, 131)
drone_execution_task_color = (255, 255, 0)
task_color = (230, 80, 80)

pg.init()
screen = pg.display.set_mode((width, height))
pg.display.set_caption("Drone Task Allocation Simulation")
font = pg.font.SysFont("consolas", 16)
clock = pg.time.Clock()

def rotate(vector, angle):
    cos_value = np.cos(angle)
    sin_value = np.sin(angle)
    return np.array([vector[0] * cos_value - vector[1] * sin_value, vector[0] * sin_value + vector[1] * cos_value])

def generate_drone_non_overlapping_position(drones):
    while True:
        position = np.array([random.uniform(margin, width - margin), random.uniform(margin, height - margin)], dtype = float)

        overlapping = False
        for drone in drones:
            if np.linalg.norm(position - drone.position) < min_drone_distance:
                overlapping = True
                break

        if not overlapping:
            return position

class Task:
    def __init__(self, position):
        self.position = position
        self.color = task_color
        self.executed = False
        self.assigned = False 

    def draw(self, surface):
        pg.draw.circle(surface, self.color, self.position.astype(int), task_radius)

class Drone:
    def __init__(self, index, position):
        self.id = index
        self.position = np.array(position, dtype = float)
        self.angle = random.uniform(0, np.pi * 2)
        self.speed = drone_speed
        self.velocity = np.array([np.cos(self.angle), np.sin(self.angle)]) * self.speed
        self.change_direction_interval = random.randint(60, 240)
        self.change_direction_timer = 0
        self.rotor_spin = random.uniform(0, np.pi * 2)
        self.rotor_speed = drone_rotor_speed
        
        temp_list = []
        for i in range(4):
            if i % 2 == 0:
                temp_list.append(1)
            else:
                temp_list.append(-1)
        self.rotor_directions = temp_list

        self.completed_tasks = []
        self.task_count = 0
        self.target_task = None

    def is_task_in_fov(self, task):
        forward = np.array([np.cos(self.angle), np.sin(self.angle)])
        difference = task.position - self.position
        distance = np.linalg.norm(difference)
        
        if distance > camera_view_distance:
            return False
            
        unit_difference = difference / distance
        dot_product = np.dot(forward, unit_difference)

        angle_to_task = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return angle_to_task <= (camera_fov_angle / 2)

    def get_neighbor_drones(self, drones):
        neighbors = []
        for drone in drones:
            if np.linalg.norm(self.position - drone.position) <= drone_communication_radius:
                neighbors.append(drone)
        return neighbors

    def update(self, drones, tasks):
        for task in tasks:
            if task.executed or task.assigned:
                continue

            if self.is_task_in_fov(task):
                if task not in self.completed_tasks:
                    self.completed_tasks.append(task)
                    
                    neighbor_drones = self.get_neighbor_drones(drones)
                    candidates = neighbor_drones + [self]
                    
                    best_drone = None
                    min_distance = float('inf')
                    for drone in candidates:
                        if drone.target_task is None:
                            distance = np.linalg.norm(drone.position - task.position)
                            if distance < min_distance:
                                min_distance = distance
                                best_drone = drone

                    if best_drone:
                        best_drone.target_task = task
                        task.assigned = True 

        separation_force = np.array([0.0, 0.0])
        for drone in drones:
            if drone is self:
                continue

            difference = self.position - drone.position
            distance = np.linalg.norm(difference)

            if distance < min_drone_distance and distance != 0:
                separation_force += difference / (distance * distance)

        if np.linalg.norm(separation_force) > 0:
            separation_force = separation_force / np.linalg.norm(separation_force) * 0.5

        if self.target_task:
            target_vector = self.target_task.position - self.position
            distance = np.linalg.norm(target_vector)
            if distance > 10:
                cruise_direction = target_vector / distance
            else:
                self.target_task.executed = True 
                self.target_task = None 
                cruise_direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        else:
            self.change_direction_timer += 1
            if self.change_direction_timer >= self.change_direction_interval:
                self.change_direction_timer = 0
                self.change_direction_interval = random.randint(60, 240)
                self.angle += random.uniform(-np.pi / 4, np.pi / 4)

            cruise_direction = np.array([np.cos(self.angle), np.sin(self.angle)])

        move_direction = cruise_direction + separation_force
        normalization = np.linalg.norm(move_direction)
        if normalization != 0:
            move_direction /= normalization

        self.velocity = move_direction * self.speed
        self.position += self.velocity

        if np.linalg.norm(self.velocity) > 0:
            self.angle = np.arctan2(self.velocity[1], self.velocity[0])

        if self.position[0] < 0: 
            self.position[0] = width
        elif self.position[0] > width: 
            self.position[0] = 0
        if self.position[1] < 0: 
            self.position[1] = height
        elif self.position[1] > height: 
            self.position[1] = 0

        self.rotor_spin += self.rotor_speed
        if self.rotor_spin > np.pi * 2:
            self.rotor_spin -= np.pi * 2

    def draw(self, surface):
        forward = np.array([np.cos(self.angle), np.sin(self.angle)])
        left_direction = rotate(forward, +camera_fov_angle / 2)
        right_direction = rotate(forward, -camera_fov_angle / 2)

        point_0 = self.position
        point_1 = self.position + left_direction * camera_view_distance
        point_2 = self.position + right_direction * camera_view_distance

        triangle_points = [(int(point_0[0]), int(point_0[1])), (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1]))]

        draw_fov_color = None
        if self.target_task:
            draw_fov_color = drone_execution_task_color
        else:
            draw_fov_color = drone_search_task_color
        pg.draw.polygon(surface, draw_fov_color, triangle_points, 1)

        offsets = [rotate(np.array([drone_arm_length, 0]), np.pi / 4), rotate(np.array([-drone_arm_length, 0]), np.pi / 4), rotate(np.array([0, drone_arm_length]), np.pi / 4), rotate(np.array([0, -drone_arm_length]), np.pi / 4)]
        
        point_1 = self.position + rotate(offsets[0], self.angle)
        point_2  = self.position + rotate(offsets[1], self.angle)
        pg.draw.line(surface, drone_body_color, point_1, point_2, 3)
        point_3 = self.position + rotate(offsets[2], self.angle)
        point_4 = self.position + rotate(offsets[3], self.angle)
        pg.draw.line(surface, drone_body_color, point_3, point_4, 3)

        for i, o in enumerate(offsets):
            rotor_position = self.position + rotate(o * 1.2, self.angle)
            pg.draw.circle(surface, drone_body_color, rotor_position, drone_rotor_protect_radius, 1)

            rotor_rotation = self.rotor_spin * self.rotor_directions[i]
            distance = rotate(np.array([drone_rotor_length, 0]), rotor_rotation)
            pg.draw.line(surface, drone_rotor_color, rotor_position - distance, rotor_position + distance, 2)

def main():
    drones = []
    tasks = []
    last_task_spawn_time = pg.time.get_ticks()

    for i in range(drone_number):
        drones.append(Drone(i, generate_drone_non_overlapping_position(drones)))

    running = True
    while running:
        current_time = pg.time.get_ticks()

        if current_time - last_task_spawn_time >= task_spawn_interval:
            last_task_spawn_time = current_time
            for _ in range(random.randint(1, 3)):
                tasks.append(Task(np.array([random.uniform(margin, width - margin), random.uniform(margin, height - margin)], dtype = float)))

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        screen.fill(background_color)

        for task in tasks:
            if not task.executed:
                task.draw(screen)

        for drone in drones:
            drone.update(drones, tasks)
            drone.draw(screen)

        temp_tasks = []
        for task in tasks:
            if not task.executed:
                temp_tasks.append(task)
        tasks = temp_tasks

        pg.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
