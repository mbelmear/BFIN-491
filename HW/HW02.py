import sys
import random

def coin_flip_game():
    # Initialize game variables
    game_count = 0
    player_wins = 0
    python_wins = 0

    def play_coin_flip():
        nonlocal game_count
        nonlocal player_wins
        nonlocal python_wins

        # Function to decide the winner of the game
        def decide_winner(user_guess, coin_flip_results):
            nonlocal player_wins
            nonlocal python_wins
            # Check if the user's guess matches any of the coin flip results
            if user_guess == coin_flip_results[0].lower() or user_guess == coin_flip_results[1].lower():
                player_wins += 1
                return "🎉 You win!"
            else:
                python_wins += 1
                return "🐍 Python wins!"

        # Get player's guess
        user_guess = input("\nEnter your guess (Head/Tail): ").strip().lower()

        # Validate player's input
        while user_guess not in ["head", "tail"]:
            print("Invalid input. Please enter either 'Head' or 'Tail'.")
            user_guess = input("\nEnter your guess (Head/Tail): ").strip().lower()

        # Simulate flipping the coin(s)
        coin_flip_results = flip_coin()

        # Print the result of the coin flip(s)
        print("Coin flip result(s):", coin_flip_results)

        # Decide the winner
        game_result = decide_winner(user_guess, coin_flip_results)

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
            play_again = input("\nY for Yes or Q to Quit\n").lower()
            if play_again not in ["y", "q"]:
                continue
            else:
                break

        # If player chooses to play again, recursively call play_coin_flip function
        if play_again == "y":
            return play_coin_flip()
        # If player chooses to quit, print goodbye message and exit
        else:
            print("\n🎉🎉🎉🎉")
            print("Thank you for playing!\n")
            sys.exit("Bye! 👋")

    return play_coin_flip

def flip_coin(num_coins=2):  # Set default number of coins to 2
    """Function to simulate flipping a coin."""
    results = [random.choice(["Head", "Tail"]) for _ in range(num_coins)]
    return results

# Call the play_coin_flip function to start the game
play = coin_flip_game()
play()