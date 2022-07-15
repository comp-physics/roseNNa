
hey = hey + RESHAPE(broadc(addLayer%adder,(/1, 5, 2, 3/),RESHAPE((/2, 5, 3, 2/),(/2, 2/), order=[2,1])), (/1, 5, 2, 3/))