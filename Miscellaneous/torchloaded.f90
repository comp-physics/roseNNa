include 'torchloader2.f90'
PROGRAM linear
USE weights_biases
!---------------
! adding any number of layers to our neural network
!----------------
IMPLICIT NONE

INTERFACE
    FUNCTION linear_layer(inp, weights) result(output)
        real :: output(size(weights, dim=1))
        real, intent(in) :: weights(:,:)
        real, intent(in) :: inp(:)

    END FUNCTION linear_layer

    FUNCTION sigmoid(x) result(output)

        real, intent(in) :: x(:)
        real :: output(size(x))

    END FUNCTION sigmoid

    FUNCTION relu(x) result(output)

        real, intent(in) :: x(:)
        real :: output(size(x))

    END FUNCTION relu

END INTERFACE

real, dimension(2,2) :: weights
real :: inp(2)
real :: out(size(weights, dim=1))

weights = reshape ( (/ 3, 2, 1, 4 /), (/ 2, 2/) )
inp = (/ -1, 1 /)

out = linear_layer(inp, weights)
print *, relu(out)
print *, sigmoid(out)
END PROGRAM linear

!=======================================================================================

FUNCTION linear_layer(inp, weights) result(output)

IMPLICIT NONE

real, intent(in) :: weights(:,:)
real, intent(in) :: inp(:)
real :: output(size(weights, dim=1))

output = matmul(weights,inp)
END FUNCTION linear_layer


FUNCTION sigmoid(x) result(output)

real, intent(in) :: x(:)
real :: output(size(x))

output = 1 / (1 + exp(-1 * x))

END FUNCTION sigmoid

FUNCTION relu(x) result(output)

real, intent(in) :: x(:)
real :: output(size(x))

where (x < 0)
    output = 0
elsewhere
    output = x
end where

END FUNCTION relu
