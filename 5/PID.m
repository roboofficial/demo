clc;
clear;

dt=0.1;
T=15;
t = 0:dt:T;
N= length(t);


figure;

path = self_pid_controller(0.8,0,0,dt,N, 3, 3);
subplot(2,2,1);
plotPath(path,"P controller",3,3);

path = self_pid_controller(0.8,0,0.6,dt,N, 3, 3);
subplot(2,2,2);
plotPath(path,"PD controller",3,3);

path = self_pid_controller(0.8,0.02,0,dt,N, 3, 3);
subplot(2,2,3);
plotPath(path,"PI controller",3,3);

path = self_pid_controller(0.8,0.02,0.6,dt,N, 3, 3);
subplot(2,2,4);
plotPath(path,"PID controller",3,3);






function path = self_pid_controller(Kp,Ki,Kd,dt,N, x_d, y_d)
    integral  = 0;
    prev_error = 0;
    path = zeros(N,3);
    path(1,:) = [0,0,0];
    for k=2:N
        error = sqrt((x_d - path(k-1,1))^2+(y_d - path(k-1,2))^2);
        integral = error * dt + integral;
        theta_d = atan2(y_d-path(k-1,2), x_d - path(k-1,1));
        derivative = (error - prev_error)/dt;
        omega = 2*(theta_d - path(k-1,3));
        v = Kp * error + Ki * integral + Kd * derivative;
        v = min(v,2);
        path(k,1) = path(k-1,1) + v*cos(path(k-1,3))*dt;
        path(k,2) = path(k-1,2) + v*sin(path(k-1,3))*dt;
        path(k,3) = path(k-1,3) + omega*dt;
        prev_error = error;
    end
end

function plotPath(path,controllerType, x_d, y_d)
X = path(:,1);
Y = path(:,2);
theta = path(end,3);
plot(X,Y,'b','LineWidth',2);
 hold on;
 %% Start Position
 plot(X(1),Y(1),  'go','MarkerSize',10,'LineWidth',2);
 %% Goal Position
 plot(x_d,y_d,  'kx','MarkerSize',12,'LineWidth',3);
 %% Final Robot Position
 plot(X(end),Y(end),  'ro','MarkerSize',10,'LineWidth',2);
 %% Robot Direction Arrow
 quiver(X(end),Y(end),  cos(theta),sin(theta),  0.8,'r','LineWidth',2);
 grid on;
 axis equal;
 xlabel('X Position');
 ylabel('Y Position');
 title([controllerType ' Controller']);
 legend('Trajectory',  'Start Position',  'Goal Position', 'Robot Position')
end
