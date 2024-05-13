
/* Routine for real variable SBX crossover */
void crossSBX(EMO_Rand *rand, double *child1, double *child2, double *parent1, double *parent2, MOP *mop, double Pc, double eta_c)
{
	double betaq;
	double y1, y2, yl, yu;
	double c1, c2;
	double rnd;
	int i;

	if (EMO_Rand_flip(rand, Pc))
	{

		for (i = 0; i < mop->nvar; i++)
		{
			//      if (EMO_Rand_flip(rand, 0.5)) {

			if (fabs(parent1[i] - parent2[i]) > _EPS)
			{

				if (parent1[i] < parent2[i])
				{
					y1 = parent1[i];
					y2 = parent2[i];
				}
				else
				{
					y1 = parent2[i];
					y2 = parent1[i];
				}

				yl = mop->xmin[i];
				yu = mop->xmax[i];
				rnd = EMO_Rand_prob3(rand);
				//          if((y2 - y1) == 0)
				//            beta = 1.0 + (2.0*(y1-yl)/1e-4);
				//          else
				//            beta = 1.0 + (2.0*(y1-yl)/(y2-y1));
				//          alpha = 2.0 - pow(beta,-(eta_c+1.0));

				//          if (rnd <= (1.0/alpha))
				//            betaq = pow ((rnd*alpha),(1.0/(eta_c+1.0)));
				//
			}
		}
	}
}

// Función que estaba mal
void mutatePolynom(EMO_Rand *rand, double *child, MOP *mop, double Pm, double eta_m)
{
	double rnd, delta1, delta2, mut_pow, deltaq;
	double y, yl, yu, val, xy;
	int j;

	for (j = 0; j < mop->nvar; j++)
	{
		if (EMO_Rand_flip(rand, Pm))
		{
			y = child[j];
			yl = mop->xmin[j];
			yu = mop->xmax[j];
			delta1 = (y - yl) / (yu - yl);
			delta2 = (yu - y) / (yu - yl);
			rnd = EMO_Rand_prob3(rand);
			mut_pow = 1.0 / (eta_m + 1.0);

			if (rnd <= 0.5)
			{
				xy = 1.0 - delta1;
				val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, (eta_m + 1.0)));
				deltaq = pow(val, mut_pow) - 1.0;
			}
			else
			{
				xy = 1.0 - delta2;
				val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, (eta_m + 1.0)));
				deltaq = 1.0 - (pow(val, mut_pow));
			}

			y = y + deltaq * (yu - yl);

			if (y < yl)
				y = yl;
			if (y > yu)
				y = yu;

			child[j] = y;
		}
	}
}

// Para imprimir población
for (int i = 0; i < N; i++)
{
	for (int j = 0; j < m; j++)
	{
		printf("%f", pop->obj[i * m + j])
	}
}
