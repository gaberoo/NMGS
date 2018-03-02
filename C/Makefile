CC     = gcc-7
CFLAGS = -g -isystem /usr/local/include -O3 -fopenmp
EFLAGS = -g -fopenmp
EFILE  = NMGS
LIBS   = -L/usr/local/lib -lm -lgsl -framework Accelerate
OBJS   = NMGS.o 

$(EFILE) : $(OBJS)
	@echo "linking..."
	$(CC) $(EFLAGS) -o $(EFILE) $(OBJS) $(LIBS)

$(OBJS) : NMGS.c NMGS.h
	$(CC) $(CFLAGS) -c $*.c 

clean:
	rm -rf *.o NMGS
