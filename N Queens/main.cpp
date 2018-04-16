#include <iostream>
#include <stdlib.h> //import for rand function
#include <time.h>   //import for time function in srand
#include <vector>   // using a vector data structure
#include <math.h>   //used for POW function
#include <time.h>   //used to generate random seed each run time (TIME)
using namespace std;

const int e = 2.71828182845;//used for simulated Annealing
//prints the solution to the terminal
void print(int *game, int gameSize)
{
    for (int i=0;i<gameSize;i++)
    {
        for (int j=0;j<gameSize;j++)
        {
            if (j==game[i])
            {
                cout<<"("<<game[i]<<","<<game[j]<<")"<<" "; //prints the "co-ordinates" of queen locations
            }
        }
    }
    cout<<endl;
}


//function to evaluate threatened queens
int checkAttack(int *game, int gameSize)
{
    int h = 0; //score heuristic
    for (int i=0;i<gameSize-1;i++)
    {
        for (int j=i+1;j<gameSize;j++) //make sure j is always above i (j=i+1)
        {
            int row_i = game[i]; // put in variables so that they can be called multiple times easier
            int row_j = game[j];

            if (row_i == row_j || // check vertical attacks
                row_i - row_j == i - j || //check diagonals
                row_i - row_j ==  j - i)
            {
                h += 1; //if a queen is threatened, increase the score
            }
        }
    }
    return h; //will need to return the score
}

//this function works similarly to checkAttack, but instead the score is decremented
int checkWeight(int *game, int gameSize){

    int h = 28;
    for (int i=0;i<gameSize-1;i++)
    {
        for (int j=i+1;j<gameSize;j++)
        {
            int row_i = game[i];
            int row_j = game[j];

            if (row_i == row_j ||
                row_i - row_j == i - j ||
                row_i - row_j ==  j - i)
            {
                h--;
            }
        }
    }
    return h;
}

void geneticSearch(int *game, int gameSize){
    double mutationProb = 0.5; //probably for mutation, higher probability = faster times
    int weightDistribute[1000] = {}; //create an empty array for the weightDistribute evaluation later on
    //WARNING - the size of weightDist needs to be larger to accommodate larger board sizes and is extremely inefficient for this problem, but it works
    int weightDistLength = 0; //int for the weightDistribute length
    int exitCode = 0;
    int pop[gameSize][gameSize]; //2d array, first parameter=population, second parameter=gameSize
    int child[gameSize][gameSize]; //both parameters are set to gameSize for simplification
    const int optimalWeight = 28;

    for(int i=0;i<gameSize;i++)
    {
        for(int j=0;j<gameSize;j++)
        {
            pop[i][j] = rand() % gameSize; //randomly populate the population
        }
    }

    while(exitCode == 0)
    {
        for(int i=0;i<gameSize;i++)
        {
            if(optimalWeight == checkWeight(pop[i], gameSize)) //if the population weight is perfect (no attacking queens) then we found a solution
            {
                print(pop[i], gameSize); //print the solution
                exitCode = 1; //exit the while loop
            }
        }
        //probability
        for(int n=0;n<gameSize;n++)
        {
            int weight = checkWeight(pop[n], gameSize); //get the weight value of population and assign to weight variable
            for(int k=0;k<weight;k++) //iterate through 'weight' amount of times, this can be completely different each runtime
            {
                weightDistribute[weightDistLength] = n; //filling weightDist with member number
                weightDistLength++;
            }
        }

        //begin reproduce
        for(int i=0;i<gameSize;i++)
        {
            int parent1 = weightDistribute[rand() % weightDistLength]; //each parent gets a randomised 0-weightDistLength
            int parent2 = weightDistribute[rand() % weightDistLength];
            int r = rand() % gameSize; //generate a random number between 0-gameSize
            for(int k=0;k<r;k++)  //begin crossover, iteration count is never known as it uses crossover probability
            {
                child[i][k] = pop[parent1][k]; //assign child both parents
                child[i+1][k] = pop[parent2][k];
            }
            for(int k=r;k<gameSize;k++)
            {
                child[i][k] = pop[parent2][k]; //reverse the order
                child[i+1][k] = pop[parent1][k];
            }
            if(mutationProb*500000>=rand() % 500000)    //begin mutation, number chosen at random (500,000)
            { //like mutation this statement should be rare to occur
                int c = rand() % 3;
                if(c == 0)
                {
                    child[i][rand() % gameSize] = rand() % gameSize; //random index gets random input
                }
                else
                {
                    child[i+1][rand() % gameSize] = rand() % gameSize; //do the same but instead increment the place
                }
            }
        }
        for(int i=0;i<gameSize;i++)
        {
            for(int j=0;j<gameSize;j++)
            {
                pop[i][j] = child[i][j]; //assigning population equal to child
            }
        }
        weightDistLength = 0; //reset back to 0 for next while loop iteration
    }
}

void simulatedAnnealing(int *game, int gameSize){

    int t = 100; //higher temperate = higher probability of replacing current solution with a worse one
    int decayRate = 0.90; //anneal rate
    int energyDiff; //delta e
    int tempGame[gameSize];
    int ran = (rand()%1000)/1000; //generate random integer between 0-999 and divide it by 1000
    for(int k = 0; k < 200000; k++) //if no solution is found within these iterations then produce nothing
        {
        t = t * decayRate; //decreases t each iteration
        for(int i=0; i<gameSize; i++)
        {
        tempGame[i] = game[i]; //give tempGame Game's contents
        tempGame[rand()%gameSize] = rand()%gameSize; //pick a random move
        energyDiff = checkAttack(tempGame, gameSize) - checkAttack(game, gameSize); //calculates the difference between the change in boards
        }
        if(ran < pow(e,-energyDiff * t) || energyDiff > 0) //if the random integer is less than  P(eD)=exp(eD*T), take the bad move
        {
            for(int i=0;i<gameSize;i++)
            {
                game[i] = tempGame[i]; //take the move
            }
        }
        if(checkAttack(game, gameSize) == NULL) //when heuristic reaches 0 print out solution (if there is one)
        {
            print(game, gameSize);
            break; //need to exit otherwise it will loop and display all possibilities
        }
    }
    cout<<endl<<"Cannot find a solution"<< endl; //if taken too long to find solution

}

int* randomRestarthillClimbing(int *game, int gameSize)
{
    //contains the best moves
    vector<int> container; //creating a vector to manipulate with vector functions(clear, push, size)
    int newhCost;
    int htoBeat = checkAttack(game, gameSize);
    int *gameOut;
    gameOut = new int [gameSize];

    for (int i=0;i<gameSize;i++)
    {
            gameOut[i] = game[i];
    }
    for (int i=0;i<gameSize;i++)
    {
        container.clear(); //removes all elements from the vector (garbage), which makes the container size 0
        container.push_back(gameOut[i]); //add elements of gameOut array to vector
        for (int k=0;k<gameSize;k++)
        {
            gameOut[i] = k; //move the queen to the new row
            newhCost = checkAttack(gameOut, gameSize); //assigning gameOut
            if (newhCost == htoBeat) //comparing if gameOut and Game are equal when input into checkAttack function
            {
                container.push_back(k); //add element k to vector (adds to the end of the vector)
            }

            else if (newhCost < htoBeat) //if it's a better move
            {
                htoBeat = newhCost;
            }
        }
        gameOut[i] = container[rand() % container.size()]; //pick a random best move
    }
    return gameOut;
}

int* hillClimbing(int *game, int gameSize)
{
    int htoBeat = checkAttack(game, gameSize);
    int *gameOut;
    gameOut = new int [gameSize];

    for (int i=0;i<gameSize;i++)
    {
            gameOut[i] = game[i];
    }
    for (int i=0;i<gameSize;i++)
    {
        for (int k=0;k<gameSize;k++)
        {
            gameOut[i] = k; //move the queen to the new row
            int newhCost = checkAttack(gameOut, gameSize); //assigning gameOut

            if (newhCost < htoBeat) // return the first better match found
            {
               game[i] = k;
            }
        }
    return game;
    }
}
bool stateNext(int *game, int gameSize)
{
    int eval = checkAttack(game, gameSize);
    int *temp;

    temp = randomRestarthillClimbing(game, gameSize);

    if (checkAttack(temp, gameSize)<eval) //if hillclimb return H is less than original game H, set game equal to hillclimb
    {
        for (int i=0;i<gameSize;i++)
        {
            game[i] = temp[i];
        }
        return  true;
    }
    return false; //if hillclimb has a larger heuristic than Game, then return false and do nothing
}



// put one queen in each row
void randomGameState(int *game, int gameSize)
{
    for (int i = 0; i < gameSize; i++)
    {
        game[i] = rand() % gameSize; //uses the rand function to help randomly input queens on each row
    }
}
//if stateNext function return NULL then reset current state
void solve(int gameSize)
{
    int count = 0;
    int *game;
    int *hillClimbTemp;
    game = new int[gameSize]; // dynamically allocates space of size gameSize, returns pointer to first element which is assigned to game
    randomGameState(game, gameSize); //generate random game state (randomly placed queen per row)
    while (checkAttack(game, gameSize) != 0) //while heuristic does not equal 0, continue looping
    {
        if (!stateNext(game, gameSize)) //if stateNext is false, reset the state
        {
            count++; //count the number of restarts
            randomGameState(game, gameSize); // reseting the state
        }
    }
        cout<<endl<<"The solution for "<<gameSize<<" queens using Hill Climbing random restart: "<<endl;
        print(game, gameSize);
        cout<<"Number of restarts: "<<count<<endl;
        cout<<endl<<"The solution for "<<gameSize<<" queens using Hill Climbing: "<<endl;
        hillClimbTemp = hillClimbing(game, gameSize);
        print(hillClimbTemp, gameSize);
        cout<<endl<<"The solution for "<<gameSize<<" queens using Genetic Algorithm: "<<endl;
        geneticSearch(game, gameSize);
        cout<<endl<<"The solution for "<<gameSize<<" queens using Simulated annealing search: "<<endl;
        simulatedAnnealing(game, gameSize);
        int test;
        cout<<"";
        cin>>test;
}

int main()
{
    srand(time(NULL));  //generate a new seed each runtime
    int m;
    cout<<"Enter integer for game size: ";
    cin>>m;
    if(m<4) //no solutions if m is less than 4
    {
        cout<<"Number needs to be at least greater than 3"<<endl;
    }
    else
    {
     solve(m);
    }
    return 0;
}
