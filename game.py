import pygame
import random


class PongGame:
    def __init__(self, screen_size=(800, 600)):
        """
        Initialize the Pong game for manual tick updates.
        """
        pygame.init()
        self.screen_width, self.screen_height = screen_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Paddle setup
        paddle_width, paddle_height = 10, 100
        paddle_x = 20
        paddle_y = (self.screen_height - paddle_height) // 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, paddle_width, paddle_height)
        # Track paddle Y as float for subpixel
        self._paddle_y = float(paddle_y)
        # Convert 5 px/frame @60fps -> px per second -> px per ms
        speed_per_sec = 5 * 60
        self.paddle_speed_ms = speed_per_sec / 1000.0

        # Ball setup
        ball_radius = 10
        # Track as floats for subpixel motion
        self.ball_x = self.screen_width / 2.0
        self.ball_y = self.screen_height / 2.0
        self.ball = pygame.Rect(
            int(self.ball_x - ball_radius),
            int(self.ball_y - ball_radius),
            ball_radius * 2,
            ball_radius * 2,
        )
        # Ball speed: 3 px/frame @60fps
        speed_per_sec_ball = 3 * 60
        self.ball_speed_x_ms = (
            speed_per_sec_ball * (1 if pygame.time.get_ticks() % 2 == 0 else -1)
        ) / 1000.0
        self.ball_speed_y_ms = (
            speed_per_sec_ball * (1 if pygame.time.get_ticks() % 3 == 0 else -1)
        ) / 1000.0

        # Monitoring variables
        self.ball_distance_x = 0
        self.ball_position = "middle"  # 'above', 'middle', 'below'
        self.paddle_hit = False
        self.ball_missed = False

        # Cooldown in ms (e.g. 1000ms = 1s)
        self.paddle_cooldown_ms = 0

        # Simulation time
        self.sim_time_ms = 0.0

        # For drawing at 60fps simulation
        self._accum_time_ms = 0.0
        self._frame_interval_ms = 1000.0 / 60.0

    def tick(self, dt_ms=1.0, move_up=False):
        """
        Advance the simulation by dt_ms milliseconds.
        move_up/move_down: control inputs for the paddle.
        """
        # Advance simulation clock
        self.sim_time_ms += dt_ms

        # Poll events (keep window responsive)
        pygame.event.pump()

        # Paddle movement via input flags
        if move_up and self._paddle_y > 0:
            self._paddle_y -= self.paddle_speed_ms * dt_ms
        if not move_up and self._paddle_y < self.screen_height - self.paddle.height:
            self._paddle_y += self.paddle_speed_ms * dt_ms
        # Apply to rect
        self.paddle.y = int(self._paddle_y)

        # Update ball position (float)
        self.ball_x += self.ball_speed_x_ms * dt_ms
        self.ball_y += self.ball_speed_y_ms * dt_ms
        # Assign to rect (center-based)
        self.ball.centerx = int(self.ball_x)
        self.ball.centery = int(self.ball_y)

        # Reset flags
        self.paddle_hit = False
        self.ball_missed = False

        # Wall collisions
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y_ms *= -1
        if self.ball.left <= 0:
            self.ball_speed_x_ms *= -1
            self.paddle_cooldown_ms = 1000
            self.ball_missed = True
        if self.ball.right >= self.screen_width:
            self.ball_speed_x_ms *= -1

        # Paddle collision if cooldown expired
        if (
            self.paddle_cooldown_ms <= 0
            and self.ball.colliderect(self.paddle)
            and self.ball_speed_x_ms < 0
        ):
            self.ball_speed_x_ms *= -1
            # Reposition
            self.ball_x = self.paddle.right + self.ball.width / 2
            self.ball.centerx = int(self.ball_x)
            self.paddle_hit = True

        # Update monitoring metrics
        self.ball_distance_x = abs(self.ball.centerx - self.paddle.centerx)

        if self.ball.centery < self.paddle.centery:
            self.ball_position = "above"
        elif self.ball.centery > self.paddle.centery:
            self.ball_position = "below"
        else:
            self.ball_position = "middle"

        # Cooldown countdown
        if self.paddle_cooldown_ms > 0:
            self.paddle_cooldown_ms -= dt_ms

        # Draw at simulated 60fps
        self._accum_time_ms += dt_ms
        if self._accum_time_ms >= self._frame_interval_ms:
            self._accum_time_ms -= self._frame_interval_ms
            self._draw()

    def _draw(self):
        """
        Internal: redraws the current frame.
        """
        self.screen.fill(self.BLACK)
        pygame.draw.rect(self.screen, self.WHITE, self.paddle)
        pygame.draw.ellipse(self.screen, self.WHITE, self.ball)
        pygame.display.flip()

    def get_simulation_time(self):
        """
        Returns the total elapsed simulation time in milliseconds.
        """
        return self.sim_time_ms

    def quit(self):
        """
        Clean up and close the game window.
        """
        pygame.quit()


# ====================================================================================================================


from model import NeuralNetwork as NN
import torch


class GameState:
    def __init__(
        self,
        num_neurons_total,
        num_neurons_train,
    ):
        self.last_time_touched_wall = -100
        self.last_time_touched_paddle = -100
        self.ball_above = True
        self.current_state = "observe"

        self.negative_time = 4
        self.positive_time = 1

        self.sim_time = 0

        self.num_neurons_total = num_neurons_total
        self.num_neurons_train = num_neurons_train

    def update_state(
        self,
        paddle_hit,
        ball_missed,
        ball_position,
        ms_passed,
        ball_dist,
    ):
        self.sim_time += ms_passed / 1000
        if paddle_hit:
            self.last_time_touched_paddle = self.sim_time
        if ball_missed:
            self.last_time_touched_wall = self.sim_time
        self.ball_above = True if ball_position == "above" else False

        if self.sim_time - self.last_time_touched_wall <= self.negative_time:
            self.current_state = "punish"
        elif self.sim_time - self.last_time_touched_paddle <= self.positive_time:
            self.current_state = "reward"
        else:
            self.current_state = "observe"

        zeros = torch.zeros(self.num_neurons_total)
        if self.current_state == "observe":
            if self.ball_above:
                zeros[: (self.num_neurons_train) // 2] = -0.000625 * ball_dist + 1
                return zeros
            else:
                zeros[(self.num_neurons_train) // 2 : self.num_neurons_train] = (
                    -0.000625 * ball_dist + 1
                )
                return zeros
        elif self.current_state == "reward":
            zeros[: self.num_neurons_train] = 2
            return zeros
        elif self.current_state == "punish":
            zeros[: self.num_neurons_train] = torch.randn(self.num_neurons_train)
            return zeros


total_neurons = 500
neurons_train = 100

nn = NN(total_neurons, timestep_duration_ms=0.1)


game_thing = GameState(
    num_neurons_total=total_neurons,
    num_neurons_train=neurons_train,
)

initial = nn.connection_strengths

# Example usage:
game = PongGame()
try:
    while True:
        # model sets move_up/move_down each tick:

        paddle_hit = game.paddle_hit
        ball_missed = game.ball_missed
        ball_position = game.ball_position
        ms_passed = nn.timestep_duration_ms
        ball_dist = game.ball_distance_x

        tensor_thing = game_thing.update_state(
            paddle_hit,
            ball_missed,
            ball_position,
            ms_passed,
            ball_dist,
        )
        up, down = nn.timed_check(raw_current=tensor_thing)
        go_up = True if up >= down else False

        game.tick(dt_ms=nn.timestep_duration_ms, move_up=go_up)
        print(
            f"time: {game_thing.sim_time:.2f}, up: {up:,}, down: {down:,}, distance: {ball_dist}, state: {game_thing.current_state}, {tensor_thing[:5]}, {tensor_thing[50:55]}"
        )

except KeyboardInterrupt:
    game.quit()

    final = nn.connection_strengths

    print(f"{final-initial}")
