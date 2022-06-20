module FILEREADER
    contains
    subroutine init(weights_0, bias_0, weights_1, bias_1, weights_2, bias_2)
        implicit none

            real :: weights_0(2,2)
            real :: bias_0(2)
            real :: weights_1(3,2)
            real :: bias_1(3)
            real :: weights_2(1,3)
            real :: bias_2(1)
        open(12, file = "weights_biases.txt")

            read(12, *), weights_0
            read(12, *), bias_0
            read(12, *), weights_1
            read(12, *), bias_1
            read(12, *), weights_2
            read(12, *), bias_2
        close(12)
    end subroutine
end module
