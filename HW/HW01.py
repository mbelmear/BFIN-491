import sys
import random
from enum import Enum

def rps():
    # Initialize game variables
    game_count = 0
    player_wins = 0
    python_wins = 0

    def play_rps():
        nonlocal game_count
        nonlocal player_wins
        nonlocal python_wins

        # Define an Enum for Rock, Paper, Scissors, Lizard, Spock
        class RPS(Enum):
            ROCK = 1
            PAPER = 2
            SCISSORS = 3
            LIZARD = 4
            SPOCK = 5

        # Define winning combinations
        winning_combinations = {
            (RPS.SCISSORS, RPS.PAPER),
            (RPS.PAPER, RPS.ROCK),
            (RPS.ROCK, RPS.LIZARD),
            (RPS.LIZARD, RPS.SPOCK),
            (RPS.SPOCK, RPS.SCISSORS),
            (RPS.SCISSORS, RPS.LIZARD),
            (RPS.LIZARD, RPS.PAPER),
            (RPS.PAPER, RPS.SPOCK),
            (RPS.SPOCK, RPS.ROCK),
            (RPS.ROCK, RPS.SCISSORS)
        }

        # Function to decide the winner of the game
        def decide_winner(player, computer):
            nonlocal player_wins
            nonlocal python_wins
            if (player, computer) in winning_combinations:
                player_wins += 1
                return "🎉 You win!"
            elif player == computer:
                return "😲 Tie game!"
            else:
                python_wins += 1
                return "🐍 Python wins!"

        # Get player's choice
        playerchoice = input(
            "\nEnter...\n1 for Rock,\n2 for Paper,\n3 for Scissors,\n4 for Lizard,\nor 5 for Spock:\n\n")

        # Validate player's input
        if playerchoice not in ["1", "2", "3", "4", "5"]:
            print("You must enter 1, 2, 3, 4, or 5.")
            return play_rps()

        # Convert player's choice to enum
        player = RPS(int(playerchoice))

        # Generate computer's choice
        computer = RPS(random.randint(1, 5))

        # Print choices made by player and computer
        print(f"\nYou chose {player.name}.")
        print(f"Python chose {computer.name}.\n")

        # Decide the winner
        game_result = decide_winner(player, computer)

        # Print the game result
        print(game_result)

        # Update game count and scores
        game_count += 1
        print(f"\nGame count: {game_count}")
        print(f"Player wins: {player_wins}")
        print(f"Python wins: {python_wins}")

        # Ask if the player wants to play again
        print("\nPlay again?")

        # Validate player's choice to play again or quit
        while True:
            playagain = input("\nY for Yes or Q to Quit\n")
            if playagain.lower() not in ["y", "q"]:
                continue
            else:
                break

        # If player chooses to play again, recursively call play_rps function
        if playagain.lower() == "y":
            return play_rps()
        # If player chooses to quit, print goodbye message and exit
        else:
            print("\n🎉🎉🎉🎉")
            print("Thank you for playing!\n")
            sys.exit("Bye! 👋")

    return play_rps

# Call the play_rps function to start the game
play = rps()
play()