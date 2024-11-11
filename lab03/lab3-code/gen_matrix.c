#include <stdio.h>
#include <stdlib.h>

void gen_matrix(char *s, int m, int n)
{
    FILE *fp; 
    if ((fp = fopen(s, "w")) == NULL)
    {
        printf("Unable to open %s for reading.\n", s);
        exit(0);
    }

    fprintf(fp, "%d\t%d", m, n);
    fprintf(fp, "\n");
    int i, j;
    for (i = 0; i < m; i++)
    {
    	for (j = 0; j < n; j++)
    	{
    		fprintf(fp, "%d\t", rand()%10);
    	}
        fprintf(fp, "\n");
    }
        
    fclose(fp);
    // printf("%lf\n", in_matrix[0]);
}

int main(int argc, char **argv)
{
	int m, n;
	char *s;
	s = argv[1];
	m = atoi(argv[2]);
	n = atoi(argv[3]);
	gen_matrix(s, m, n);
	
	return 0;
}
