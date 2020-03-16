module Dynamics

import Base
using DifferentialEquations
import LinearAlgebra.eigvals
import LinearAlgebra.Symmetric
import LinearAlgebra.qr
import LinearAlgebra.diag
import LinearAlgebra.diagm
import LinearAlgebra.dot
import LinearAlgebra.norm
import LinearAlgebra.I
import LinearAlgebra.mul!
import Statistics.mean

export to_unit, gen_J0_eigs, gen_J0_constrained_eigs
export gen_GOE_mats, gen_GOE_mats_set_e1, piece_lin
export getFinalPoint, getFinalPointStochastic, getTrajectoryStochastic

function to_unit(v)
	return v/norm(v)
end
function rotation_matrix(v1,v2) #finds rotation matrix between two (normalized) vectors (rotates v1 to v2)
    u = to_unit(v1)
    v = to_unit(v2)
    costh = dot(u,v)
    #sinth = np.sin(np.arccos(costh))
    sinth = sqrt(1-costh^2)
    R = [costh -sinth ;; sinth costh]
    w = (v - dot(u,v)*u); w = to_unit(w)
    uw_mat = hcat(u,w)'
    return I - u*u' - w*w' + (uw_mat')*R*uw_mat
end
function gen_J0_eigs(n)
	R = Base.randn(Float64, (n, n))
	J = (R + transpose(R))/sqrt(2*n)
	J = Symmetric(J)
	return sort(eigvals(J),rev=true)
end
function gen_J0_constrained_eigs(n,condition; n_tries=100, kwargs... )
	for i in range(1,length=n_tries)
		J0eigs = gen_J0_eigs(n)
		if condition(J0eigs[1])
			return J0eigs
		end
	end
	error(string("could not satisfy eigenvalue condition after ",n_tries, " tries"))
	return J0eigs
end
function gen_haar_mat(n)
	rand_mat = Base.randn(Float64,(n,n))
    qr_result = qr(rand_mat)

    #protip: make Q and R unique.
    L = diagm(0 => diag(qr_result.R) ./ abs.(diag(qr_result.R))) #transform to unique Q', R'
    return qr_result.Q * L
end

function gen_GOE_mats_set_e1(e1;kwargs...)
	n = length(e1)
	_, Jeigs, Jvecs = gen_GOE_mats(n;kwargs...)
	rot = rotation_matrix(Jvecs[:,1],e1)
    Jvecs = rot*Jvecs
	J = Jvecs * diagm(0 => Jeigs) * transpose(Jvecs)
	return J, Jeigs, Jvecs
end
function gen_GOE_mats(n;eig_cond = eig -> true, kwargs... )
	Jeigs = gen_J0_constrained_eigs(n,eig_cond; kwargs... )
	Jvecs = gen_haar_mat(n)
	J = Jvecs * diagm(0 => Jeigs) * transpose(Jvecs)
	return J, Jeigs, Jvecs
end

function piece_lin(x)
	if abs(x) < 1
		return x
	else
		return sign(x)
	end
end
#functions are inplace: we assign by .= as to not waste memory
#also these functions don't work properly if we assign dx = instead of dx .= 

#return function with selected nonlinearity
function make_xdot_no_input(phi)
	function xdot_no_input(dx,x,p,t)
		J = p[1]
		tanh_mult = p[2]
		dx .= -x + J*phi.(tanh_mult*x)
		#mul!(dx,J,phi.(tanh_mult*x))
		#dx .= -x + dx
	end
	return xdot_no_input
end

function make_xdot_pulse_input(phi)
	function xdot_pulse_input(dx,x,p,t)
		J = p[1]
		tanh_mult = p[2]
		input = p[3]
		pulse_len = p[4]
		if t <= pulse_len
			dx .= -x + J*phi.(tanh_mult*x) + input
			#mul!(dx,J,phi.(tanh_mult*x))
			#dx .= -x + dx + input
		else
			dx .= -x + J*phi.(tanh_mult*x)
			#mul!(dx,J,phi.(tanh_mult*x))
			#dx .= -x + dx
		end
	end
	return xdot_pulse_input
end

function getFinalPoint(J,x0,input,T,pulse_len,tanh_mult;
	phi = tanh,
	reltol=1e-8, kwargs... )
	xdot_no_input = make_xdot_no_input(phi)
	xdot_pulse_input = make_xdot_pulse_input(phi)
	if isapprox(T,0)
		return input
	end
	tspan = (0.0,T)
	if isapprox(pulse_len,0)
		prob = ODEProblem(xdot_no_input,x0,tspan,(J,tanh_mult))
	else
		prob = ODEProblem(xdot_pulse_input,x0,tspan,(J,tanh_mult,input,pulse_len))
	end
	sol = solve(prob,reltol=reltol,save_everystep=false)
	return sol.u[end]
end

function getFinalPointStochastic(J,x0,input,T,pulse_len,sigma_t,tanh_mult;
	phi = tanh,
	reltol=1e-8, kwargs... )
	xdot_no_input = make_xdot_no_input(phi)
	xdot_pulse_input = make_xdot_pulse_input(phi)
	if isapprox(T,0)
		return input
	end
	function noise_fun(dx2,x,p,t)
		return dx2 .= sigma_t
	end
	tspan = (0.0,T)
	if isapprox(pulse_len,0)
		prob = SDEProblem(xdot_no_input,noise_fun,x0,tspan,(J,tanh_mult))
	else
		prob = SDEProblem(xdot_pulse_input,noise_fun,x0,tspan,(J,tanh_mult,input,pulse_len))
	end
	sol = solve(prob,reltol=reltol,save_everystep=false,dense=false)
	return sol.u[end]
end

function getTrajectoryStochastic(J,x0,input,tspan,pulse_len,sigma_t,tanh_mult;
	phi = tanh,
	reltol=1e-8, kwargs... )
	xdot_no_input = make_xdot_no_input(phi)
	xdot_pulse_input = make_xdot_pulse_input(phi)

	function noise_fun(dx2,x,p,t)
		return dx2 .= sigma_t
	end

	if isapprox(pulse_len,0)
		prob = SDEProblem(xdot_no_input,noise_fun,x0,tspan,(J,tanh_mult))
	else
		prob = SDEProblem(xdot_pulse_input,noise_fun,x0,tspan,(J,tanh_mult,input,pulse_len))
	end
	sol = solve(prob,reltol=reltol,save_everystep=true;kwargs...)
	return sol
end

end
