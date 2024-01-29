import math

"""A pseudo-random number generator (PRNG) to replace the default one provided by Numpy.
    This algorithm was obtained from a programming manual for the HP-15C scientific calculator,
    c. 1978, and has been shown to reliably produce a uniformly random sequence of values
    (warning not tested to the level required for security applications).  Evaluations were
    made per "The Art of Computer Programming, Vol 2, Seminumerical Methods" by Donald Knuth.
"""

class HpPrng:

    _seed = 0.0     #seed value is always in [0, 1)

    def __init__(self,
                 seed : int = 0   #if provided, its value must be > 0
                ):
        """Initializes the PRNG object, possibly with a non-negative integer seed.  If None is specified for
            the seed, then the seed will be set to 0.
        """

        if isinstance(seed, int)  and  seed >= 0:
            HpPrng._seed = float(seed) / (math.pow(2, 32) - 1.0) #Python can handle larger values, but this is good
        elif seed != None:
            raise TypeError("HpPrng requires an integer seed >= 0. Given type & value was: {}, {}".format(type(seed), seed))

        self.counter = 0


    def random(self) -> float:
        """Returns a pseudo-random value in [0, 1)"""

        self.counter += 1
        irn = self._new_irn()

        # Pull out the second digit of the irn; if it is a particular value, do an additional draw. This helps to prevent
        # a correlated sequence.
        div10 = irn//10
        div100 = irn//100
        dig2 = div10 - 10*div100
        if dig2 == 3:
            self._new_irn()
        return self._seed


    def gaussian(self,
                 mean   : float = 0.0,  #mean value of the distribution
                 stddev : float = 1.0   #standard deviation of the distribution
                ) -> float:
        """Returns a pseudo-random value from a Guassian distribution with mean and stddev."""

        sum = 0.0
        for _ in range(6):
            sum += self.random()

        return stddev*(sum - 3.0) + mean


    def _new_irn(self) -> int:
        """Returns a new integer rn and sets a new seed."""

        rn = 9821.0*(self._seed + 0.211327)
        irn = int(rn)
        self._seed = rn - irn

        return irn

    def _get_seed(self):
        """FOR TESTING ONLY!"""
        return self._seed


    def uses_since(self) -> int:
        """Returns the count of calls to random() since initializing or since the previous call to this method."""

        c = self.counter
        self.counter = 0
        return c
