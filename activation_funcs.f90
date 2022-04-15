module activation_functions
    implicit none
    
contains

    FUNCTION sigmoid(x) result(output)
        real, intent(in) :: x(:)
        real :: output(size(x))
        
        output = 1 / (1 + exp(-1 * x))
    END FUNCTION sigmoid

    FUNCTION sigmoid2d(x) result(output)
        real, intent(in) :: x(:,:)
        real :: output(size(x,1), size(x,2))
        
        output = 1 / (1 + exp(-1 * x))
    END FUNCTION sigmoid2d
    
    FUNCTION relu(x) result(output)
        real, intent(in) :: x(:)
        real :: output(size(x))
        
        where (x < 0)
            output = 0
        elsewhere
            output = x
        end where
    END FUNCTION relu

    FUNCTION tanhh(x) result(output)
        real, intent(in) :: x(:)
        real :: output(size(x))
        output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
    END FUNCTION tanhh

    FUNCTION tanhh2d(x) result(output)
        real, intent(in) :: x(:,:)
        real :: output(size(x,1),size(x,2))
        output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
    END FUNCTION tanhh2d
    
end module activation_functions