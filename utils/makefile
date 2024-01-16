libTM: cool_tm.c cool_tm.h 
	gcc -O3 -fopenmp -ffast-math -c -Wall -Werror -fpic cool_tm.c cool_tm.h
	gcc -shared -O3 -fopenmp -ffast-math -o libTM.so cool_tm.o

clean:
	rm *.o libTM.so
