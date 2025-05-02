import pygame
import threading
import random


class PongGame:
    def __init__(self):
        """Initialize the Pong game in a separate thread."""
        self.thread = threading.Thread(target=self.run_game, daemon=True)
        self.running = True

        # Monitoring variables
        self.ball_distance_x = 0
        self.ball_position = "middle"  # 'above', 'middle', 'below'
        self.paddle_hit = False
        self.ball_missed = False

        # Add cooldown timer
        self.paddle_cooldown = 0  # Countdown frames

        # Start the game thread
        self.thread.start()

    def run_game(self):
        """Main game loop running in a separate thread."""
        pygame.init()
        screen_width, screen_height = 800, 600
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)

        # Paddle setup
        paddle_width, paddle_height = 10, 100
        paddle_x = 20
        paddle_y = (screen_height - paddle_height) // 2
        self.paddle = pygame.Rect(paddle_x, paddle_y, paddle_width, paddle_height)
        paddle_speed = 5

        # Ball setup
        ball_radius = 10
        self.ball = pygame.Rect(
            screen_width // 2, screen_height // 2, ball_radius * 2, ball_radius * 2
        )
        ball_speed_x = 3 * random.choice([-1, 1])
        ball_speed_y = 3 * random.choice([-1, 1])

        # Miss detection state
        self.previous_miss = False

        while self.running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Paddle movement (keyboard for testing)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and self.paddle.top > 0:
                self.paddle.y -= paddle_speed
            if keys[pygame.K_DOWN] and self.paddle.bottom < screen_height:
                self.paddle.y += paddle_speed

            # Update ball position
            self.ball.x += ball_speed_x
            self.ball.y += ball_speed_y

            self.ball_missed = False

            # Wall collision (all four walls)
            if self.ball.top <= 0 or self.ball.bottom >= screen_height:
                ball_speed_y *= -1
            if self.ball.left <= 0:
                ball_speed_x *= -1
                self.paddle_cooldown = 60  # 1 second cooldown at 60 FPS
                self.ball_missed = True
            if self.ball.right >= screen_width:
                ball_speed_x *= -1

            # Paddle collision (only if not in cooldown)
            self.paddle_hit = False
            if self.paddle_cooldown <= 0 and self.ball.colliderect(self.paddle) and ball_speed_x < 0:
                ball_speed_x *= -1
                self.ball.left = self.paddle.right  # Prevent sticking
                self.paddle_hit = True

            # Monitor ball-paddle relationship
            paddle_center_x, ball_center_x = self.paddle.centerx, self.ball.centerx
            self.ball_distance_x = abs(ball_center_x - paddle_center_x)

            paddle_center_y, ball_center_y = self.paddle.centery, self.ball.centery
            if ball_center_y < paddle_center_y:
                self.ball_position = "above"
            elif ball_center_y > paddle_center_y:
                self.ball_position = "below"
            else:
                self.ball_position = "middle"

            # Update paddle cooldown
            if self.paddle_cooldown > 0:
                self.paddle_cooldown -= 1

            # Drawing
            screen.fill(BLACK)
            pygame.draw.rect(screen, WHITE, self.paddle)
            pygame.draw.ellipse(screen, WHITE, self.ball)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


# Example usage: Print monitoring data in main thread
if __name__ == "__main__":
    game = PongGame()
    try:
        while True:
            print(
                f"Distance X: {game.ball_distance_x}, Position: {game.ball_position}, "
                f"Hit: {game.paddle_hit}, Missed: {game.ball_missed}"
            )
    except KeyboardInterrupt:
        print("Stopping game...")
        game.running = False
