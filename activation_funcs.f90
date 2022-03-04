module activation_functions
    implicit none
    
contains

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

    FUNCTION tanh(x) result(output)
        real, intent(in) :: x(:)
        real :: output(size(x))
        output = 2*sigmoid(2*x)-1
    END FUNCTION tanh
    
end module activation_functions