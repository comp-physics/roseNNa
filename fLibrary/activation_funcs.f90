module activation_functions
    use iso_c_binding
    implicit none
    
contains

    FUNCTION sigmoid(x) result(output)
        REAL (c_double), intent(in) :: x(:)
        REAL (c_double) :: output(size(x))
        
        output = 1 / (1 + exp(-1 * x))
    END FUNCTION sigmoid

    FUNCTION sigmoid2d(x) result(output)
        REAL (c_double), intent(in) :: x(:,:)
        REAL (c_double) :: output(size(x,1), size(x,2))
        
        output = 1 / (1 + exp(-1 * x))
    END FUNCTION sigmoid2d
    
    FUNCTION relu(x) result(output)
        REAL (c_double), intent(in) :: x(:)
        REAL (c_double) :: output(size(x))
        
        where (x < 0)
            output = 0
        elsewhere
            output = x
        end where
    END FUNCTION relu

    FUNCTION relu2d(x) result(output)
        REAL (c_double), intent(in) :: x(:,:)
        REAL (c_double) :: output(size(x,1), size(x,2))
        
        where (x < 0)
            output = 0
        elsewhere
            output = x
        end where
    END FUNCTION relu2d

    FUNCTION relu4d(x) result(output)
        REAL (c_double), intent(in) :: x(:,:,:,:)
        REAL (c_double) :: output(size(x,1), size(x,2), size(x,3), size(x,4))
        
        where (x < 0)
            output = 0
        elsewhere
            output = x
        end where
    END FUNCTION relu4d

    FUNCTION tanhh(x) result(output)
        REAL (c_double), intent(in) :: x(:)
        REAL (c_double) :: output(size(x))
        output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
    END FUNCTION tanhh

    FUNCTION tanhh2d(x) result(output)
        REAL (c_double), intent(in) :: x(:,:)
        REAL (c_double) :: output(size(x,1),size(x,2))
        output = (exp(x)-exp(-1*x))/(exp(x)+exp(-1*x))
    END FUNCTION tanhh2d
    
end module activation_functions